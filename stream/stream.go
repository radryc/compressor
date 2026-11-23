package stream

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"runtime"
	"sync"

	"github.com/radryc/compressor/codec"
	"github.com/radryc/compressor/model"
)

// Stream layout (version 3)
//
//	header (32 bytes)
//	  [0:4]   magic "NGR3"
//	  [4]     version
//	  [5]     flags: bit0 session, bits1-3 blend
//	  [6:8]   header length
//	  [8:12]  model id
//	  [12:16] model content hash (crc32c of the artifact)
//	  [16:20] block size
//	  [20:24] frame payload cap
//	  [24:28] dictionary hash (0 = none)
//	  [28:32] crc32c of [0:28]
//
//	frame header (24 bytes) + payload
//	  [0:8]   sequence number
//	  [8:12]  raw length
//	  [12:16] compressed length
//	  [16]    flags: 0x01 passthrough, 0x02 keyframe, 0x04 dictionary, 0x80 end
//	  [17:20] reserved
//	  [20:24] crc32c of [0:20] + payload
//
// Every length is validated against hard caps before allocating, so a corrupt
// or hostile stream can never trigger an unbounded allocation.

// Magic identifies an NGR3 stream.
var Magic = [4]byte{'N', 'G', 'R', '3'}

// Version is the on-wire stream format version.
const Version = 6

const (
	// FlagPassthrough marks a frame whose payload is the raw bytes.
	FlagPassthrough uint8 = 1 << 0
	// FlagKeyframe marks a frame that carries serialised codec state.
	FlagKeyframe uint8 = 1 << 1
	// FlagDictionary marks the priming frame at the start of a stream.
	FlagDictionary uint8 = 1 << 2
	// FlagEnd marks the terminating frame.
	FlagEnd uint8 = 0x80

	flagSession = 1 << 0
	blendShift  = 1
)

const (
	headerLen = 32
	frameLen  = 24

	// MaxBlockSize bounds a single block.
	MaxBlockSize = 16 << 20
	// MaxFramePayload bounds a single frame payload.
	MaxFramePayload = 64 << 20
	// maxExpansion is the worst-case arithmetic-coder expansion per input
	// byte (a bit costs at most 16 bits with a Q16 probability).
	maxExpansion = 16
	// MemoryBudget bounds the worst-case per-process working set; the
	// compressor lowers its worker count to stay inside it.
	MemoryBudget = 1 << 30 // 1 GiB
	// maxKeyframe bounds a serialised codec snapshot.
	maxKeyframe = 1 << 20
)

// Mode selects the framing strategy.
type Mode int

const (
	// ModeBlocks compresses independent blocks in parallel: high throughput
	// and independently decodable blocks.
	ModeBlocks Mode = iota
	// ModeSession keeps model state across frames and emits keyframes. It is
	// serial but compresses many small batches far better.
	ModeSession
)

// ModelProvider resolves active models for compression and historical models
// (by id and content hash) for decompression.
type ModelProvider interface {
	Active(dataset string) (*model.Model, bool)
	Get(ctx context.Context, id, hash uint32) (*model.Model, error)
}

// Observer receives per-frame telemetry for drift detection or metrics.
type Observer interface {
	RecordBlock(dataset string, modelID uint32, rawBytes, compBytes int)
}

// Stats summarises a compression or decompression run.
type Stats struct {
	RawBytes  int64
	CompBytes int64
	Frames    int64
}

// Ratio returns compressed/raw.
func (s Stats) Ratio() float64 {
	if s.RawBytes == 0 {
		return 0
	}
	return float64(s.CompBytes) / float64(s.RawBytes)
}

// BPB returns bits per raw byte, including framing overhead.
func (s Stats) BPB() float64 {
	if s.RawBytes == 0 {
		return 0
	}
	return float64(s.CompBytes) * 8 / float64(s.RawBytes)
}

type Option func(*config)

type config struct {
	observer   Observer
	blockSize  int
	workers    int
	mode       Mode
	frameMax   int
	keyframeIP int
	dict       []byte
	blend      codec.Blend
	startSeq   uint64
}

func defaultConfig() config {
	w := runtime.GOMAXPROCS(0)
	if w < 1 {
		w = 1
	}
	return config{
		blockSize:  1 << 20,
		workers:    w,
		mode:       ModeBlocks,
		frameMax:   MaxFramePayload,
		keyframeIP: 8 << 20,
		blend:      codec.Full,
	}
}

// WithBlockSize sets the block (or session frame) size.
func WithBlockSize(n int) Option { return func(c *config) { c.blockSize = n } }

// WithWorkers sets the number of parallel block coders.
func WithWorkers(n int) Option { return func(c *config) { c.workers = n } }

// WithObserver installs per-frame telemetry.
func WithObserver(o Observer) Option { return func(c *config) { c.observer = o } }

// WithMode selects blocks or session framing.
func WithMode(m Mode) Option { return func(c *config) { c.mode = m } }

// WithDictionary primes both sides with raw bytes that are not part of the
// output, the way a zstd dictionary works.
func WithDictionary(d []byte) Option { return func(c *config) { c.dict = d } }

// WithBlend selects the predictors (used by the ablation study).
func WithBlend(b codec.Blend) Option { return func(c *config) { c.blend = b } }

// WithKeyframeInterval sets how many raw bytes a session stream may code
// before emitting a resumable keyframe.
func WithKeyframeInterval(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.keyframeIP = n
		}
	}
}

// WithStartSeq numbers the first frame n, and on the decompression side skips
// frames before n.
func WithStartSeq(n uint64) Option { return func(c *config) { c.startSeq = n } }

func (c *config) validate() error {
	if c.blockSize <= 0 || c.blockSize > MaxBlockSize {
		return fmt.Errorf("block size %d out of range (1..%d)", c.blockSize, MaxBlockSize)
	}
	if c.frameMax <= 0 || c.frameMax > MaxFramePayload {
		return fmt.Errorf("frame cap %d out of range (1..%d)", c.frameMax, MaxFramePayload)
	}
	if len(c.dict) > c.blockSize {
		return fmt.Errorf("dictionary %d bytes exceeds block size %d", len(c.dict), c.blockSize)
	}
	if c.workers < 1 {
		c.workers = 1
	}
	return nil
}

func dictHash(d []byte) uint32 {
	if len(d) == 0 {
		return 0
	}
	return crc32.Checksum(d, crc32.MakeTable(crc32.Castagnoli))
}

// ==========================================
// Header
// ==========================================

type header struct {
	flags     uint8
	modelID   uint32
	modelHash uint32
	blockSize uint32
	frameMax  uint32
	dictHash  uint32
}

func (h *header) marshal() [headerLen]byte {
	var b [headerLen]byte
	copy(b[0:4], Magic[:])
	b[4] = Version
	b[5] = h.flags
	binary.LittleEndian.PutUint16(b[6:8], headerLen)
	binary.LittleEndian.PutUint32(b[8:12], h.modelID)
	binary.LittleEndian.PutUint32(b[12:16], h.modelHash)
	binary.LittleEndian.PutUint32(b[16:20], h.blockSize)
	binary.LittleEndian.PutUint32(b[20:24], h.frameMax)
	binary.LittleEndian.PutUint32(b[24:28], h.dictHash)
	binary.LittleEndian.PutUint32(b[28:32], crc32.Checksum(b[0:28], crc32.MakeTable(crc32.Castagnoli)))
	return b
}

func unmarshalHeader(b []byte) (*header, error) {
	if len(b) < headerLen {
		return nil, errors.New("stream: short header")
	}
	if !bytes.Equal(b[0:4], Magic[:]) {
		return nil, errors.New("stream: bad magic (expected NGR3)")
	}
	if b[4] != Version {
		return nil, fmt.Errorf("stream: unsupported version %d", b[4])
	}
	if binary.LittleEndian.Uint16(b[6:8]) != headerLen {
		return nil, errors.New("stream: bad header length")
	}
	if got, want := crc32.Checksum(b[0:28], crc32.MakeTable(crc32.Castagnoli)), binary.LittleEndian.Uint32(b[28:32]); got != want {
		return nil, errors.New("stream: header checksum mismatch")
	}
	h := &header{
		flags:     b[5],
		modelID:   binary.LittleEndian.Uint32(b[8:12]),
		modelHash: binary.LittleEndian.Uint32(b[12:16]),
		blockSize: binary.LittleEndian.Uint32(b[16:20]),
		frameMax:  binary.LittleEndian.Uint32(b[20:24]),
		dictHash:  binary.LittleEndian.Uint32(b[24:28]),
	}
	if h.blockSize == 0 || h.blockSize > MaxBlockSize {
		return nil, fmt.Errorf("stream: block size %d out of range", h.blockSize)
	}
	if h.frameMax == 0 || h.frameMax > MaxFramePayload {
		return nil, fmt.Errorf("stream: frame cap %d out of range", h.frameMax)
	}
	return h, nil
}

func (h *header) blend() codec.Blend {
	f := (h.flags >> blendShift) & 0x0f
	return codec.Blend{GRU: f&1 != 0, Match: f&2 != 0, Table: f&4 != 0, Run: f&8 != 0}
}

func (h *header) session() bool { return h.flags&flagSession != 0 }

// ==========================================
// Frames
// ==========================================

type frameHeader struct {
	seq       uint64
	rawLen    uint32
	compLen   uint32
	flags     uint8
	profile   uint8 // low nibble: codec.Profile; high nibble: codec.Transform
	transform uint8 // mirror of the high nibble, filled by parse
	stride    uint16
	crc       uint32
}

func (f *frameHeader) marshal() [frameLen]byte {
	var b [frameLen]byte
	binary.LittleEndian.PutUint64(b[0:8], f.seq)
	binary.LittleEndian.PutUint32(b[8:12], f.rawLen)
	binary.LittleEndian.PutUint32(b[12:16], f.compLen)
	b[16] = f.flags
	b[17] = f.profile | f.transform<<4
	binary.LittleEndian.PutUint16(b[18:20], f.stride)
	binary.LittleEndian.PutUint32(b[20:24], f.crc)
	return b
}

func parseFrameHeader(b []byte) *frameHeader {
	return &frameHeader{
		seq:       binary.LittleEndian.Uint64(b[0:8]),
		rawLen:    binary.LittleEndian.Uint32(b[8:12]),
		compLen:   binary.LittleEndian.Uint32(b[12:16]),
		flags:     b[16],
		profile:   b[17] & 0x0f,
		transform: b[17] >> 4,
		stride:    binary.LittleEndian.Uint16(b[18:20]),
		crc:       binary.LittleEndian.Uint32(b[20:24]),
	}
}

// frameCRC covers the mutable part of the frame header plus the payload.
func frameCRC(fh *frameHeader, payload []byte) uint32 {
	var tmp [20]byte
	binary.LittleEndian.PutUint64(tmp[0:8], fh.seq)
	binary.LittleEndian.PutUint32(tmp[8:12], fh.rawLen)
	binary.LittleEndian.PutUint32(tmp[12:16], fh.compLen)
	tmp[16] = fh.flags
	tmp[17] = fh.profile | fh.transform<<4
	binary.LittleEndian.PutUint16(tmp[18:20], fh.stride)
	h := crc32.New(crc32.MakeTable(crc32.Castagnoli))
	h.Write(tmp[:20])
	h.Write(payload)
	return h.Sum32()
}

// validateFrame enforces structural invariants and allocation caps.
func validateFrame(h *header, fh *frameHeader) error {
	if fh.flags&FlagEnd != 0 {
		if fh.compLen != 0 || fh.rawLen != 0 {
			return errors.New("stream: end frame carries data")
		}
		return nil
	}
	if fh.compLen > h.frameMax {
		return fmt.Errorf("stream: frame payload %d exceeds cap %d", fh.compLen, h.frameMax)
	}
	if !codec.Profile(fh.profile).Valid() {
		return fmt.Errorf("stream: unknown profile %d", fh.profile)
	}
	if !codec.Transform(fh.transform).Valid() {
		return fmt.Errorf("stream: unknown transform %d", fh.transform)
	}
	if fh.stride != 0 && (codec.Profile(fh.profile) != codec.ProfileImage || uint32(fh.stride) > fh.rawLen) {
		return fmt.Errorf("stream: invalid stride %d for profile %d", fh.stride, fh.profile)
	}
	if fh.transform != byte(codec.TransformNone) && (codec.Profile(fh.profile) != codec.ProfileImage || fh.stride == 0) {
		return fmt.Errorf("stream: transform %d needs the image profile and a stride", fh.transform)
	}
	if fh.flags&FlagPassthrough != 0 {
		if fh.compLen != fh.rawLen {
			return fmt.Errorf("stream: passthrough length mismatch (%d != %d)", fh.compLen, fh.rawLen)
		}
		if fh.stride != 0 {
			return errors.New("stream: passthrough frame carries a transform")
		}
		if fh.rawLen > h.blockSize {
			return fmt.Errorf("stream: raw length %d exceeds block size", fh.rawLen)
		}
		return nil
	}
	if fh.flags&FlagKeyframe != 0 {
		if fh.rawLen != 0 {
			return fmt.Errorf("stream: keyframe with raw length %d", fh.rawLen)
		}
		if fh.compLen > maxKeyframe {
			return fmt.Errorf("stream: keyframe too large (%d)", fh.compLen)
		}
		return nil
	}
	if fh.flags&FlagDictionary != 0 {
		if fh.rawLen != 0 || fh.compLen > h.blockSize {
			return errors.New("stream: malformed dictionary frame")
		}
		return nil
	}
	if fh.rawLen > h.blockSize {
		return fmt.Errorf("stream: raw length %d exceeds block size %d", fh.rawLen, h.blockSize)
	}
	if uint64(fh.compLen) > uint64(fh.rawLen)*maxExpansion+8 {
		return fmt.Errorf("stream: implausible coded length %d for %d raw bytes", fh.compLen, fh.rawLen)
	}
	return nil
}

func writeFrame(w io.Writer, fh *frameHeader, payload []byte) error {
	fh.crc = frameCRC(fh, payload)
	b := fh.marshal()
	if _, err := w.Write(b[:]); err != nil {
		return err
	}
	if len(payload) > 0 {
		_, err := w.Write(payload)
		return err
	}
	return nil
}

// readFrame reads one frame, validating structure and checksum.
func readFrame(r io.Reader, h *header) (*frameHeader, []byte, error) {
	var hb [frameLen]byte
	if _, err := io.ReadFull(r, hb[:]); err != nil {
		if err == io.EOF {
			return nil, nil, io.ErrUnexpectedEOF
		}
		return nil, nil, fmt.Errorf("stream: read frame header: %w", err)
	}
	fh := parseFrameHeader(hb[:])
	if err := validateFrame(h, fh); err != nil {
		return nil, nil, err
	}
	var payload []byte
	if fh.compLen > 0 {
		payload = make([]byte, fh.compLen)
		if _, err := io.ReadFull(r, payload); err != nil {
			return nil, nil, fmt.Errorf("stream: read frame payload: %w", err)
		}
	}
	if fh.flags&FlagEnd == 0 {
		if got := frameCRC(fh, payload); got != fh.crc {
			return nil, nil, fmt.Errorf("stream: frame %d checksum mismatch (corrupt or truncated)", fh.seq)
		}
	}
	return fh, payload, nil
}

// ==========================================
// Compressor
// ==========================================

// Compressor writes NGR3 streams.
type Compressor struct {
	models ModelProvider
	cfg    config
}

func NewCompressor(models ModelProvider, opts ...Option) (*Compressor, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	// Memory guard. Table sizes are a function of the profile (so the decoder
	// always agrees), therefore the only safe knob is the worker count: clamp
	// it so the worst-case working set stays inside the budget.
	if cfg.mode == ModeBlocks {
		per := codec.ConfigForProfile(codec.ProfileText, cfg.blockSize).EngineBytes()
		if per > 0 {
			maxWorkers := int(MemoryBudget / per)
			if maxWorkers < 1 {
				maxWorkers = 1
			}
			if cfg.workers > maxWorkers {
				cfg.workers = maxWorkers
			}
		}
	}
	return &Compressor{models: models, cfg: cfg}, nil
}

func (c *config) blendFlags() uint8 {
	var f uint8
	if c.blend.GRU {
		f |= 1
	}
	if c.blend.Match {
		f |= 2
	}
	if c.blend.Table {
		f |= 4
	}
	if c.blend.Run {
		f |= 8
	}
	return f << blendShift
}

func (c *config) codecConfig() codec.Config {
	return c.codecConfigFor(codec.ProfileTelemetry)
}

func (c *config) codecConfigFor(p codec.Profile) codec.Config {
	cc := codec.ConfigForProfile(p, c.blockSize)
	cc.Blend = c.blend
	if c.mode == ModeSession {
		cc.MaxHist = 4 << 20
		cc.TableBits = 18
	}
	return cc
}

// Compress reads r to EOF and writes a complete NGR3 stream to w.
func (c *Compressor) Compress(ctx context.Context, dataset string, r io.Reader, w io.Writer) (Stats, error) {
	m, ok := c.models.Active(dataset)
	if !ok {
		return Stats{}, fmt.Errorf("no active model registered for dataset %q", dataset)
	}
	h := &header{
		flags:     c.cfg.blendFlags(),
		modelID:   m.ID,
		modelHash: m.Hash,
		blockSize: uint32(c.cfg.blockSize),
		frameMax:  uint32(c.cfg.frameMax),
		dictHash:  dictHash(c.cfg.dict),
	}
	if c.cfg.mode == ModeSession {
		h.flags |= flagSession
	}
	hb := h.marshal()
	if _, err := w.Write(hb[:]); err != nil {
		return Stats{}, err
	}

	st := &Stats{}
	seq := c.cfg.startSeq
	if len(c.cfg.dict) > 0 {
		if err := writeFrame(w, &frameHeader{seq: seq, compLen: uint32(len(c.cfg.dict)), flags: FlagDictionary}, c.cfg.dict); err != nil {
			return *st, err
		}
		seq++
	}
	var err error
	if c.cfg.mode == ModeSession {
		seq, err = c.compressSession(ctx, dataset, m, r, w, seq, st)
	} else {
		seq, err = c.compressBlocks(ctx, dataset, m, r, w, seq, st)
	}
	if err != nil {
		return *st, err
	}
	return *st, writeFrame(w, &frameHeader{seq: seq, flags: FlagEnd}, nil)
}

func (c *Compressor) blockWindow() int {
	n := 4 * c.cfg.workers
	if n < 8 {
		n = 8
	}
	return n
}

type compJob struct {
	seq  int
	data []byte
}

type compResult struct {
	seq         int
	origLen     int
	payload     []byte
	passthrough bool
	profile     codec.Profile
	transform   codec.Transform
	stride      int
}

func (c *Compressor) compressBlocks(ctx context.Context, dataset string, m *model.Model, r io.Reader, w io.Writer, seq0 uint64, st *Stats) (uint64, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan compJob, c.cfg.workers)
	results := make(chan compResult, c.cfg.workers*2)
	sem := make(chan struct{}, c.blockWindow())

	var wg sync.WaitGroup
	for i := 0; i < c.cfg.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var engines [3]*codec.BlockEngine
			engine := func(p codec.Profile) *codec.BlockEngine {
				if engines[p] == nil {
					engines[p] = codec.NewBlockEngine(m, c.cfg.codecConfigFor(p))
				}
				return engines[p]
			}
			for job := range jobs {
				d := Classify(job.data)
				if d.Transform != codec.TransformNone {
					d.Transform.Apply(job.data, d.Stride)
				}
				dst := make([]byte, 0, len(job.data))
				out, pt := engine(d.Profile).CompressBlockWithDict(m, c.cfg.dict, job.data, dst)
				stride, tr := d.Stride, d.Transform
				if pt {
					// The payload is the original bytes: undo the transform so
					// the frame is a plain passthrough.
					if tr != codec.TransformNone {
						tr.Undo(out, stride)
						tr, stride = codec.TransformNone, 0
					}
				}
				select {
				case results <- compResult{seq: job.seq, origLen: len(job.data), payload: out, passthrough: pt, profile: d.Profile, transform: tr, stride: stride}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	readErr := make(chan error, 1)
	go func() {
		defer close(jobs)
		seq := 0
		for {
			if err := ctx.Err(); err != nil {
				readErr <- err
				return
			}
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				readErr <- ctx.Err()
				return
			}
			buf := make([]byte, c.cfg.blockSize)
			n, err := io.ReadFull(r, buf)
			if n > 0 {
				select {
				case jobs <- compJob{seq: seq, data: buf[:n]}:
					seq++
				case <-ctx.Done():
					readErr <- ctx.Err()
					return
				}
			}
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				readErr <- nil
				return
			}
			if err != nil {
				readErr <- err
				return
			}
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	pending := make(map[int]compResult)
	next := 0
	var writeErr error
	seq := seq0

	for res := range results {
		if writeErr != nil {
			continue
		}
		pending[res.seq] = res
		for {
			ready, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			<-sem
			next++

			fh := &frameHeader{
				seq: seq, rawLen: uint32(ready.origLen), compLen: uint32(len(ready.payload)),
				profile: byte(ready.profile), transform: byte(ready.transform), stride: uint16(ready.stride),
			}
			if ready.passthrough {
				fh.flags |= FlagPassthrough
			}
			if err := writeFrame(w, fh, ready.payload); err != nil {
				writeErr = err
				cancel()
				break
			}
			seq++
			st.RawBytes += int64(ready.origLen)
			st.CompBytes += int64(frameLen + len(ready.payload))
			st.Frames++
			if c.cfg.observer != nil {
				c.cfg.observer.RecordBlock(dataset, m.ID, ready.origLen, len(ready.payload))
			}
		}
	}

	if err := <-readErr; err != nil {
		return seq, err
	}
	return seq, writeErr
}

func (c *Compressor) compressSession(ctx context.Context, dataset string, m *model.Model, r io.Reader, w io.Writer, seq uint64, st *Stats) (uint64, error) {
	prof := codec.ProfileTelemetry
	tr := codec.TransformNone
	stride := 0
	eng := codec.NewBlockEngine(m, c.cfg.codecConfigFor(prof))
	eng.Reset(m)
	if len(c.cfg.dict) > 0 {
		eng.Prime(c.cfg.dict)
	}
	sinceKeyframe := 0
	codedSinceKF := 0
	kfInterval := c.cfg.keyframeIP
	buf := make([]byte, c.cfg.blockSize)
	for {
		if err := ctx.Err(); err != nil {
			return seq, err
		}
		n, err := io.ReadFull(r, buf)
		if n > 0 {
			chunk := buf[:n]
			// A keyframe is only worth writing when it costs at most a tenth
			// of the bytes coded since the previous one. On a very
			// compressible stream that can mean very few keyframes, which is
			// fine: such a stream is cheap to replay from the start.
			if sinceKeyframe >= kfInterval && len(eng.History()) >= codec.KeyframeHistory {
				histLen := len(eng.History())
				if histLen > codec.KeyframeHistory {
					histLen = codec.KeyframeHistory
				}
				if codec.SnapshotSize(m.HiddenDim, histLen)*10 <= codedSinceKF {
					eng.TrimHistory(codec.KeyframeHistory)
					snap := eng.Snapshot(nil)
					if e := writeFrame(w, &frameHeader{seq: seq, compLen: uint32(len(snap)), flags: FlagKeyframe}, snap); e != nil {
						return seq, e
					}
					seq++
					sinceKeyframe = 0
					codedSinceKF = 0
				}
			}
			work := chunk
			if tr != codec.TransformNone {
				tr.Apply(work, stride)
			}
			out := eng.CompressFrame(work, nil)
			fh := &frameHeader{
				seq: seq, rawLen: uint32(n), compLen: uint32(len(out)),
				profile: byte(prof), transform: byte(tr), stride: uint16(stride),
			}
			payload := out
			if len(out) >= n {
				fh.flags |= FlagPassthrough
				fh.compLen = uint32(n)
				if tr != codec.TransformNone {
					tr.Undo(work, stride)
					fh.transform = byte(codec.TransformNone)
					fh.stride = 0
				}
				payload = work
			}
			if e := writeFrame(w, fh, payload); e != nil {
				return seq, e
			}
			seq++
			sinceKeyframe += n
			codedSinceKF += len(payload)
			st.RawBytes += int64(n)
			st.CompBytes += int64(frameLen + len(payload))
			st.Frames++
			if c.cfg.observer != nil {
				c.cfg.observer.RecordBlock(dataset, m.ID, n, len(payload))
			}
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return seq, nil
		}
		if err != nil {
			return seq, err
		}
	}
}

// ==========================================
// Decompressor
// ==========================================

// Decompressor reads NGR3 streams.
type Decompressor struct {
	models ModelProvider
	cfg    config
}

func NewDecompressor(models ModelProvider, opts ...Option) (*Decompressor, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &Decompressor{models: models, cfg: cfg}, nil
}

// Decompress reads an NGR3 stream from r and writes the original bytes to w.
func (d *Decompressor) Decompress(ctx context.Context, r io.Reader, w io.Writer) (Stats, error) {
	var hb [headerLen]byte
	if _, err := io.ReadFull(r, hb[:]); err != nil {
		return Stats{}, fmt.Errorf("stream: read header: %w", err)
	}
	h, err := unmarshalHeader(hb[:])
	if err != nil {
		return Stats{}, err
	}
	m, err := d.models.Get(ctx, h.modelID, h.modelHash)
	if err != nil {
		return Stats{}, err
	}

	st := &Stats{}
	cc := d.profileConfig(h, codec.ProfileTelemetry)

	// A dictionary frame, when present, always precedes the data.
	var dict []byte
	fh, payload, err := readFrame(r, h)
	if err != nil {
		return *st, err
	}
	if fh.flags&FlagDictionary != 0 {
		dict = payload
		if h.dictHash != 0 && dictHash(dict) != h.dictHash {
			return *st, errors.New("stream: dictionary checksum mismatch")
		}
		fh, payload, err = readFrame(r, h)
		if err != nil {
			return *st, err
		}
	} else if h.dictHash != 0 {
		return *st, errors.New("stream: header declares a dictionary that is not present")
	}

	if h.session() {
		return *st, d.decompressSession(ctx, h, m, cc, dict, fh, payload, r, w, st)
	}
	return *st, d.decompressBlocks(ctx, h, m, cc, dict, fh, payload, r, w, st)
}

func (d *Decompressor) decompressBlocks(ctx context.Context, h *header, m *model.Model, cc codec.Config, dict []byte,
	first *frameHeader, firstPayload []byte, r io.Reader, w io.Writer, st *Stats) error {

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	type job struct {
		seq       uint64
		rawLen    int
		payload   []byte
		pass      bool
		profile   codec.Profile
		transform codec.Transform
		stride    int
	}
	type result struct {
		seq  uint64
		data []byte
	}

	jobs := make(chan job, d.cfg.workers)
	results := make(chan result, d.cfg.workers*2)
	sem := make(chan struct{}, d.blockWindow())

	var wg sync.WaitGroup
	for i := 0; i < d.cfg.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var engines [3]*codec.BlockEngine
			engine := func(p codec.Profile) *codec.BlockEngine {
				if engines[p] == nil {
					engines[p] = codec.NewBlockEngine(m, d.profileConfig(h, p))
				}
				return engines[p]
			}
			for j := range jobs {
				var data []byte
				if j.pass {
					data = j.payload
				} else {
					data = engine(j.profile).DecompressBlockWithDict(m, dict, j.payload, j.rawLen, make([]byte, 0, j.rawLen))
				}
				if j.transform != codec.TransformNone {
					j.transform.Undo(data, j.stride)
				}
				select {
				case results <- result{seq: j.seq, data: data}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	readErr := make(chan error, 1)
	go func() {
		defer close(jobs)
		fh, payload, err := first, firstPayload, error(nil)
		for {
			if err != nil {
				readErr <- err
				return
			}
			if fh.flags&FlagEnd != 0 {
				readErr <- nil
				return
			}
			if err := ctx.Err(); err != nil {
				readErr <- err
				return
			}
			if fh.flags&FlagKeyframe != 0 {
				fh, payload, err = readFrame(r, h)
				continue
			}
			if fh.seq < d.cfg.startSeq {
				fh, payload, err = readFrame(r, h)
				continue
			}
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				readErr <- ctx.Err()
				return
			}
			j := job{
				seq: fh.seq, rawLen: int(fh.rawLen), payload: payload,
				pass:    fh.flags&FlagPassthrough != 0,
				profile: codec.Profile(fh.profile), transform: codec.Transform(fh.transform), stride: int(fh.stride),
			}
			select {
			case jobs <- j:
			case <-ctx.Done():
				readErr <- ctx.Err()
				return
			}
			fh, payload, err = readFrame(r, h)
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	pending := make(map[uint64][]byte)
	// The first data frame sets the expected sequence; non-data frames
	// (dictionary, keyframe) consume sequence numbers too.
	next := d.cfg.startSeq
	if first != nil && first.seq > next {
		next = first.seq
	}
	var writeErr error
	for res := range results {
		if writeErr != nil {
			continue
		}
		pending[res.seq] = res.data
		for {
			data, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			<-sem
			next++
			if _, err := w.Write(data); err != nil {
				writeErr = err
				cancel()
				break
			}
			st.RawBytes += int64(len(data))
			st.Frames++
		}
	}
	if err := <-readErr; err != nil {
		return err
	}
	return writeErr
}

func (d *Decompressor) decompressSession(ctx context.Context, h *header, m *model.Model, cc codec.Config, dict []byte,
	first *frameHeader, firstPayload []byte, r io.Reader, w io.Writer, st *Stats) error {

	if first != nil {
		cc = d.profileConfig(h, codec.Profile(first.profile))
	}
	eng := codec.NewBlockEngine(m, cc)
	eng.Reset(m)
	if len(dict) > 0 {
		eng.Prime(dict)
	}
	restored := len(dict) == 0 && d.cfg.startSeq == 0

	fh, payload := first, firstPayload
	for {
		if fh.flags&FlagEnd != 0 {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if fh.flags&FlagDictionary != 0 {
			// Already primed; a second dictionary frame is malformed.
			return errors.New("stream: duplicate dictionary frame")
		}
		if fh.flags&FlagKeyframe != 0 {
			if err := eng.Restore(m, payload); err != nil {
				return err
			}
			restored = true
		} else if fh.seq >= d.cfg.startSeq {
			if !restored {
				return fmt.Errorf("stream: session resume needs a keyframe at or before frame %d", fh.seq)
			}
			var out []byte
			if fh.flags&FlagPassthrough != 0 {
				out = payload
				eng.Prime(payload)
			} else {
				out = eng.DecompressFrame(payload, int(fh.rawLen), make([]byte, 0, fh.rawLen))
				if tr := codec.Transform(fh.transform); tr != codec.TransformNone {
					tr.Undo(out, int(fh.stride))
				}
			}
			if _, err := w.Write(out); err != nil {
				return err
			}
			st.RawBytes += int64(len(out))
			st.Frames++
		}
		var err error
		fh, payload, err = readFrame(r, h)
		if err != nil {
			return err
		}
	}
}

// profileConfig derives the codec configuration for a profile from the stream
// header. Table sizes must come from the profile, never from the decoder's own
// options, or a stream written with deep tables would fail to decode.
func (d *Decompressor) profileConfig(h *header, p codec.Profile) codec.Config {
	cc := codec.ConfigForProfile(p, int(h.blockSize))
	cc.Blend = h.blend()
	if h.session() {
		cc.MaxHist = 4 << 20
		cc.TableBits = 18
	}
	return cc
}

func (d *Decompressor) blockWindow() int {
	n := 4 * d.cfg.workers
	if n < 8 {
		n = 8
	}
	return n
}
