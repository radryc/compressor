package stream

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/radryc/compressor/codec"
	"github.com/radryc/compressor/model"
)

// Writer is an io.WriteCloser that turns a byte stream into NGR3 frames.
//
// It is the streaming entry point: a pusher can hand it one OTLP batch at a
// time and call Flush, instead of waiting for 1 MiB of input. In ModeSession
// the model state carries across Flush calls, which is what makes many small
// batches compress well.
type Writer struct {
	cfg     config
	dataset string
	m       *model.Model
	w       io.Writer
	engines [3]*codec.BlockEngine
	eng     *codec.BlockEngine

	buf     []byte
	seq     uint64
	st      Stats
	started bool
	closed  bool
	sinceKF int
	codedKF int
	kfIP    int
	err     error
}

// NewWriter resolves the dataset's active model and writes the stream header.
func NewWriter(w io.Writer, models ModelProvider, dataset string, opts ...Option) (*Writer, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	m, ok := models.Active(dataset)
	if !ok {
		return nil, fmt.Errorf("no active model registered for dataset %q", dataset)
	}
	h := &header{
		flags:     cfg.blendFlags(),
		modelID:   m.ID,
		modelHash: m.Hash,
		blockSize: uint32(cfg.blockSize),
		frameMax:  uint32(cfg.frameMax),
		dictHash:  dictHash(cfg.dict),
	}
	if cfg.mode == ModeSession {
		h.flags |= flagSession
	}
	hb := h.marshal()
	if _, err := w.Write(hb[:]); err != nil {
		return nil, err
	}
	sw := &Writer{cfg: cfg, dataset: dataset, m: m, w: w}
	sw.eng = codec.NewBlockEngine(m, cfg.codecConfig())
	sw.engines[codec.ProfileTelemetry] = sw.eng
	sw.eng.Reset(m)
	if len(cfg.dict) > 0 {
		sw.eng.Prime(cfg.dict)
		seq := uint64(0)
		if err := writeFrame(w, &frameHeader{seq: seq, compLen: uint32(len(cfg.dict)), flags: FlagDictionary}, cfg.dict); err != nil {
			return nil, err
		}
		sw.seq++
	}
	sw.buf = make([]byte, 0, cfg.blockSize)
	sw.kfIP = cfg.keyframeIP
	return sw, nil
}

// Write buffers p and emits complete frames as needed.
func (w *Writer) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	if w.closed {
		return 0, errors.New("stream: write after close")
	}
	written := 0
	for len(p) > 0 {
		n := copy(w.buf[len(w.buf):cap(w.buf)], p)
		w.buf = w.buf[:len(w.buf)+n]
		p = p[n:]
		written += n
		if len(w.buf) == cap(w.buf) {
			if err := w.flushFrame(); err != nil {
				w.err = err
				return written, err
			}
		}
	}
	return written, nil
}

// Flush emits the currently buffered bytes as one frame.
func (w *Writer) Flush() error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return errors.New("stream: flush after close")
	}
	if len(w.buf) == 0 {
		return nil
	}
	return w.flushFrame()
}

// Close flushes and writes the terminating frame.
func (w *Writer) Close() error {
	if w.closed {
		return w.err
	}
	if w.err != nil {
		w.closed = true
		return w.err
	}
	if err := w.Flush(); err != nil {
		w.closed = true
		w.err = err
		return err
	}
	w.closed = true
	w.err = writeFrame(w.w, &frameHeader{seq: w.seq, flags: FlagEnd}, nil)
	return w.err
}

// Stats returns the running totals.
func (w *Writer) Stats() Stats { return w.st }

func (w *Writer) engine(p codec.Profile) *codec.BlockEngine {
	if w.engines[p] == nil {
		w.engines[p] = codec.NewBlockEngine(w.m, w.cfg.codecConfigFor(p))
		if len(w.cfg.dict) > 0 {
			w.engines[p].Prime(w.cfg.dict)
		}
	}
	if w.cfg.mode == ModeSession {
		// Session state must stay in one engine.
		return w.eng
	}
	return w.engines[p]
}

func (w *Writer) flushFrame() error {
	if len(w.buf) == 0 {
		return nil
	}
	chunk := w.buf
	if w.cfg.mode == ModeSession && w.sinceKF >= w.kfIP && len(w.eng.History()) >= codec.KeyframeHistory {
		histLen := len(w.eng.History())
		if histLen > codec.KeyframeHistory {
			histLen = codec.KeyframeHistory
		}
		if codec.SnapshotSize(w.m.HiddenDim, histLen)*10 <= w.codedKF {
			w.eng.TrimHistory(codec.KeyframeHistory)
			snap := w.eng.Snapshot(nil)
			if err := writeFrame(w.w, &frameHeader{seq: w.seq, compLen: uint32(len(snap)), flags: FlagKeyframe}, snap); err != nil {
				return err
			}
			w.seq++
			w.sinceKF = 0
			w.codedKF = 0
		}
	}

	d := Decision{Profile: codec.Profile(w.eng.Profile())}
	if w.cfg.mode == ModeBlocks {
		d = Classify(chunk)
	}
	if d.Transform != codec.TransformNone {
		d.Transform.Apply(chunk, d.Stride)
	}
	eng := w.engine(d.Profile)
	out := eng.CompressFrame(chunk, nil)
	fh := &frameHeader{
		seq: w.seq, rawLen: uint32(len(chunk)), compLen: uint32(len(out)),
		profile: byte(d.Profile), transform: byte(d.Transform), stride: uint16(d.Stride),
	}
	payload := out
	if len(out) >= len(chunk) {
		fh.flags |= FlagPassthrough
		fh.compLen = uint32(len(chunk))
		if d.Transform != codec.TransformNone {
			d.Transform.Undo(chunk, d.Stride)
			fh.transform = byte(codec.TransformNone)
			fh.stride = 0
		}
		payload = chunk
	}
	if err := writeFrame(w.w, fh, payload); err != nil {
		return err
	}
	w.seq++
	if w.cfg.mode == ModeSession {
		w.sinceKF += len(chunk)
		w.codedKF += len(payload)
	} else {
		eng.Reset(w.m)
		if len(w.cfg.dict) > 0 {
			eng.Prime(w.cfg.dict)
		}
	}
	w.st.RawBytes += int64(len(chunk))
	w.st.CompBytes += int64(frameLen + len(payload))
	w.st.Frames++
	if w.cfg.observer != nil {
		w.cfg.observer.RecordBlock(w.dataset, w.m.ID, len(chunk), len(payload))
	}
	w.buf = w.buf[:0]
	return nil
}

// Reader is an io.Reader over decompressed stream bytes. It decodes one frame
// at a time, so memory stays bounded regardless of stream length.
type Reader struct {
	cfg config
	h   *header
	m   *model.Model
	cc  codec.Config
	r   io.Reader
	eng *codec.BlockEngine

	engines [3]*codec.BlockEngine

	dict    []byte
	out     []byte
	seq     uint64
	next    uint64
	done    bool
	started bool
	st      Stats
}

// NewReader reads the stream header and resolves the model.
func NewReader(r io.Reader, models ModelProvider, opts ...Option) (*Reader, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	var hb [headerLen]byte
	if _, err := io.ReadFull(r, hb[:]); err != nil {
		return nil, fmt.Errorf("stream: read header: %w", err)
	}
	h, err := unmarshalHeader(hb[:])
	if err != nil {
		return nil, err
	}
	m, err := models.Get(context.Background(), h.modelID, h.modelHash)
	if err != nil {
		return nil, err
	}
	cc := codec.ConfigForProfile(codec.ProfileTelemetry, int(h.blockSize))
	cc.Blend = h.blend()
	if h.session() {
		cc.MaxHist = 4 << 20
		cc.TableBits = 18
	}
	sr := &Reader{cfg: cfg, h: h, m: m, cc: cc, r: r, next: cfg.startSeq}
	sr.eng = codec.NewBlockEngine(m, cc)
	sr.engines[codec.ProfileTelemetry] = sr.eng
	return sr, nil
}

func (r *Reader) Read(p []byte) (int, error) {
	for len(r.out) == 0 {
		if r.done {
			return 0, io.EOF
		}
		if err := r.fill(); err != nil {
			return 0, err
		}
	}
	n := copy(p, r.out)
	r.out = r.out[n:]
	return n, nil
}

func (r *Reader) fill() error {
	for {
		fh, payload, err := readFrame(r.r, r.h)
		if err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) {
				return io.ErrUnexpectedEOF
			}
			return err
		}
		if fh.flags&FlagEnd != 0 {
			r.done = true
			return nil
		}
		if fh.flags&FlagDictionary != 0 {
			if r.started {
				return errors.New("stream: duplicate dictionary frame")
			}
			r.dict = payload
			if r.h.dictHash != 0 && dictHash(payload) != r.h.dictHash {
				return errors.New("stream: dictionary checksum mismatch")
			}
			r.eng.Reset(r.m)
			r.eng.Prime(payload)
			continue
		}
		if fh.flags&FlagKeyframe != 0 {
			if err := r.eng.Restore(r.m, payload); err != nil {
				return err
			}
			r.started = true
			continue
		}
		if fh.seq < r.next {
			continue
		}
		if r.h.session() {
			if !r.started && len(r.dict) == 0 && r.cfg.startSeq == 0 {
				r.started = true
			}
			if !r.started {
				return fmt.Errorf("stream: session resume needs a keyframe at or before frame %d", fh.seq)
			}
		}
		prof := codec.Profile(fh.profile)
		eng := r.engine(prof)
		var data []byte
		if fh.flags&FlagPassthrough != 0 {
			data = payload
			eng.Prime(payload)
		} else {
			data = eng.DecompressFrame(payload, int(fh.rawLen), make([]byte, 0, fh.rawLen))
			if tr := codec.Transform(fh.transform); tr != codec.TransformNone {
				tr.Undo(data, int(fh.stride))
			}
		}
		if !r.h.session() {
			eng.Reset(r.m)
			if len(r.dict) > 0 {
				eng.Prime(r.dict)
			}
		}
		r.started = true
		r.out = data
		r.st.RawBytes += int64(len(data))
		r.st.Frames++
		return nil
	}
}

// Stats returns the running totals.
func (r *Reader) Stats() Stats { return r.st }

func (r *Reader) engine(p codec.Profile) *codec.BlockEngine {
	if r.h.session() {
		return r.eng
	}
	if r.engines[p] == nil {
		c := codec.ConfigForProfile(p, int(r.h.blockSize))
		c.Blend = r.h.blend()
		if r.h.session() {
			c.MaxHist = 4 << 20
			c.TableBits = 18
		}
		e := codec.NewBlockEngine(r.m, c)
		if len(r.dict) > 0 {
			e.Prime(r.dict)
		}
		r.engines[p] = e
	}
	return r.engines[p]
}
