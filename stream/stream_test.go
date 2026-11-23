package stream

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math/rand"
	"testing"

	"github.com/radryc/compressor/model"
)

type testProvider struct {
	m *model.Model
}

func (p *testProvider) Active(string) (*model.Model, bool) { return p.m, true }
func (p *testProvider) Get(_ context.Context, id, hash uint32) (*model.Model, error) {
	if id != p.m.ID {
		return nil, errors.New("model not found")
	}
	if hash != 0 && hash != p.m.Hash {
		return nil, errors.New("hash mismatch")
	}
	return p.m, nil
}

func testModel(t *testing.T, h, id int) *model.Model {
	t.Helper()
	w := model.NewWeights(h, id)
	rng := rand.New(rand.NewSource(int64(h*100 + id)))
	for _, s := range [][]float32{w.InProj, w.WHH, w.BHH, w.WBit, w.BBit} {
		for i := range s {
			s[i] = (rng.Float32()*2 - 1) * 0.25
		}
	}
	raw, err := w.Encode()
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func corpus(n int) []byte {
	line := []byte(`{"traceId":"4bf92f3577b34da6a3ce929d0e0e4736","spanId":"00f067aa0ba902b7","name":"POST /api/v1/orders","kind":2,"startTimeUnixNano":"1732347421104829000"}` + "\n")
	out := make([]byte, 0, n)
	rng := rand.New(rand.NewSource(5))
	for len(out) < n {
		c := append([]byte(nil), line...)
		for i := 0; i < 3; i++ {
			p := 12 + rng.Intn(70)
			if c[p] >= '0' && c[p] <= '9' {
				c[p] = byte('0' + rng.Intn(10))
			}
		}
		out = append(out, c...)
	}
	return out[:n]
}

func roundTrip(t *testing.T, m *model.Model, src []byte, copts, dopts []Option) Stats {
	t.Helper()
	c, err := NewCompressor(&testProvider{m}, copts...)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	st, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if st.Frames == 0 && len(src) > 0 {
		t.Fatal("no frames written")
	}
	d, err := NewDecompressor(&testProvider{m}, dopts...)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if _, err := d.Decompress(context.Background(), bytes.NewReader(buf.Bytes()), &out); err != nil {
		t.Fatalf("decompress: %v", err)
	}
	if !bytes.Equal(out.Bytes(), src) {
		t.Fatalf("roundtrip mismatch: got %d bytes want %d", out.Len(), len(src))
	}
	return st
}

func TestBlocksRoundTrip(t *testing.T) {
	m := testModel(t, 16, 7)
	for _, size := range []int{0, 1, 2, 100, 4096} {
		for _, bs := range []int{1024, 65536} {
			for _, workers := range []int{1, 4} {
				src := corpus(size)
				st := roundTrip(t, m, src, []Option{WithBlockSize(bs), WithWorkers(workers)}, nil)
				if size > 1000 {
					t.Logf("size=%-6d bs=%-6d w=%d -> %d bytes (%.3f, %.2f bpb)",
						size, bs, workers, st.CompBytes, st.Ratio(), st.BPB())
				}
			}
		}
	}
}

func TestBlocksDeterministic(t *testing.T) {
	m := testModel(t, 16, 8)
	src := corpus(300000)
	var a, b bytes.Buffer
	c1, _ := NewCompressor(&testProvider{m}, WithBlockSize(4096), WithWorkers(1))
	if _, err := c1.Compress(context.Background(), "d", bytes.NewReader(src), &a); err != nil {
		t.Fatal(err)
	}
	c2, _ := NewCompressor(&testProvider{m}, WithBlockSize(4096), WithWorkers(7))
	if _, err := c2.Compress(context.Background(), "d", bytes.NewReader(src), &b); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("output depends on worker count")
	}
}

func TestSessionRoundTrip(t *testing.T) {
	m := testModel(t, 16, 9)
	src := corpus(600000)
	st := roundTrip(t, m, src,
		[]Option{WithMode(ModeSession), WithBlockSize(8192), WithKeyframeInterval(65536)},
		nil)
	t.Logf("session: %d -> %d bytes (%.3f, %.2f bpb), %d frames", len(src), st.CompBytes, st.Ratio(), st.BPB(), st.Frames)
}

func TestSessionIsBetterThanBlocksOnSmallBatches(t *testing.T) {
	m := testModel(t, 16, 10)
	// Many small, similar batches: a session should exploit cross-batch
	// redundancy better than independent blocks.
	src := corpus(64 * 2000)
	bst := roundTrip(t, m, src, []Option{WithBlockSize(2000), WithWorkers(4)}, nil)
	sst := roundTrip(t, m, src,
		[]Option{WithMode(ModeSession), WithBlockSize(2000), WithKeyframeInterval(1 << 30)}, nil)
	t.Logf("blocks: %d bytes (%.3f) | session: %d bytes (%.3f)", bst.CompBytes, bst.Ratio(), sst.CompBytes, sst.Ratio())
}

func TestDictionaryRoundTrip(t *testing.T) {
	m := testModel(t, 16, 11)
	dict := corpus(8192)
	src := corpus(50000)
	st := roundTrip(t, m, src, []Option{WithDictionary(dict), WithBlockSize(16384)}, nil)
	t.Logf("dict: %d -> %d bytes (%.3f)", len(src), st.CompBytes, st.Ratio())
}

func TestSessionResumeFromKeyframe(t *testing.T) {
	m := testModel(t, 16, 12)
	// Keyframes are only written when they are cheap relative to the bytes
	// stored between them, so use incompressible data and a large stream.
	src := make([]byte, 3<<20)
	rand.New(rand.NewSource(19)).Read(src)
	var buf bytes.Buffer
	c, _ := NewCompressor(&testProvider{m}, WithMode(ModeSession), WithBlockSize(32768), WithKeyframeInterval(1<<20), WithWorkers(1))
	if _, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf); err != nil {
		t.Fatal(err)
	}
	// Find the second keyframe's sequence and start there.
	h, _ := unmarshalHeader(buf.Bytes()[:headerLen])
	rd := bytes.NewReader(buf.Bytes()[headerLen:])
	var kf []uint64
	for {
		fh, _, err := readFrame(rd, h)
		if err != nil || fh.flags&FlagEnd != 0 {
			break
		}
		if fh.flags&FlagKeyframe != 0 {
			kf = append(kf, fh.seq)
		}
	}
	if len(kf) < 2 {
		t.Fatalf("expected multiple keyframes, got %v", kf)
	}
	start := kf[1]
	d, _ := NewDecompressor(&testProvider{m}, WithStartSeq(start), WithWorkers(1))
	var out bytes.Buffer
	if _, err := d.Decompress(context.Background(), bytes.NewReader(buf.Bytes()), &out); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if out.Len() == 0 || out.Len() >= len(src) {
		t.Fatalf("resume produced %d bytes", out.Len())
	}
	if !bytes.Equal(out.Bytes(), src[len(src)-out.Len():]) {
		t.Fatal("resumed output does not match the tail of the source")
	}
	t.Logf("resumed at seq %d, %d bytes", start, out.Len())
}

// ---------------------------------------------------------------- robustness

func TestCorruptionDetected(t *testing.T) {
	m := testModel(t, 16, 13)
	src := corpus(100000)
	var buf bytes.Buffer
	c, _ := NewCompressor(&testProvider{m}, WithBlockSize(8192), WithWorkers(1))
	if _, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf); err != nil {
		t.Fatal(err)
	}
	good := buf.Bytes()

	// Flip a byte in the middle of the payload.
	bad := append([]byte(nil), good...)
	bad[len(bad)/2] ^= 0xFF
	var out bytes.Buffer
	if _, err := mustDecomp(t, &testProvider{m}).Decompress(context.Background(), bytes.NewReader(bad), &out); err == nil {
		t.Fatal("corruption not detected")
	}

	// Flip a header byte.
	bad2 := append([]byte(nil), good...)
	bad2[12] ^= 0xFF // model hash
	out.Reset()
	if _, err := mustDecomp(t, &testProvider{m}).Decompress(context.Background(), bytes.NewReader(bad2), &out); err == nil {
		t.Fatal("header corruption not detected")
	}
}

func TestTruncationDetected(t *testing.T) {
	m := testModel(t, 16, 14)
	src := corpus(50000)
	var buf bytes.Buffer
	c, _ := NewCompressor(&testProvider{m}, WithBlockSize(8192), WithWorkers(1))
	if _, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf); err != nil {
		t.Fatal(err)
	}
	good := buf.Bytes()
	for _, cut := range []int{1, 5, headerLen - 1, headerLen, headerLen + 3, len(good) / 2, len(good) - 1} {
		var out bytes.Buffer
		if _, err := mustDecomp(t, &testProvider{m}).Decompress(context.Background(), bytes.NewReader(good[:cut]), &out); err == nil {
			t.Errorf("cut=%d: truncation not detected (wrote %d bytes)", cut, out.Len())
		}
	}
}

func TestHostileLengthsRejected(t *testing.T) {
	m := testModel(t, 16, 15)
	mk := func(blockSize, frameMax uint32) []byte {
		h := &header{flags: 7 << blendShift, modelID: m.ID, modelHash: m.Hash, blockSize: blockSize, frameMax: frameMax}
		b := h.marshal()
		return b[:]
	}
	cases := map[string][]byte{
		"huge block size": mk(0xFFFFFFFF, MaxFramePayload),
		"huge frame cap":  mk(1<<20, 0xFFFFFFFF),
		"zero block size": mk(0, MaxFramePayload),
	}
	for name, hb := range cases {
		var out bytes.Buffer
		if _, err := mustDecomp(t, &testProvider{m}).Decompress(context.Background(), bytes.NewReader(hb), &out); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}

	// A well-formed header followed by a frame claiming 4 GiB.
	hb := mk(1<<20, MaxFramePayload)
	var fh [frameLen]byte
	binary.LittleEndian.PutUint64(fh[0:8], 0)
	binary.LittleEndian.PutUint32(fh[8:12], 1<<20)
	binary.LittleEndian.PutUint32(fh[12:16], 0xFFFFFFFF)
	stream := append(append([]byte(nil), hb...), fh[:]...)
	var out bytes.Buffer
	if _, err := mustDecomp(t, &testProvider{m}).Decompress(context.Background(), bytes.NewReader(stream), &out); err == nil {
		t.Error("overflowing frame length accepted")
	}
}

func TestCancelStopsWork(t *testing.T) {
	m := testModel(t, 16, 16)
	src := corpus(2 << 20)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c, _ := NewCompressor(&testProvider{m}, WithBlockSize(4096), WithWorkers(4))
	var buf bytes.Buffer
	if _, err := c.Compress(ctx, "d", bytes.NewReader(src), &buf); err == nil {
		t.Fatal("cancelled context did not fail")
	}
}

func TestObserverAndStats(t *testing.T) {
	m := testModel(t, 16, 17)
	src := corpus(40000)
	obs := &countingObserver{}
	c, _ := NewCompressor(&testProvider{m}, WithBlockSize(8192), WithObserver(obs))
	var buf bytes.Buffer
	st, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if obs.calls != int(st.Frames) || obs.raw != int(st.RawBytes) {
		t.Fatalf("observer saw %d calls/%d raw, stats %d/%d", obs.calls, obs.raw, st.Frames, st.RawBytes)
	}
	if st.CompBytes+headerLen+frameLen > int64(buf.Len()) {
		t.Fatalf("stats undercount: %d vs %d", st.CompBytes+headerLen+frameLen, buf.Len())
	}
}

type countingObserver struct {
	calls int
	raw   int
}

func (o *countingObserver) RecordBlock(_ string, _ uint32, raw, _ int) {
	o.calls++
	o.raw += raw
}

var _ io.Reader = (*bytes.Reader)(nil)

func mustDecomp(t *testing.T, p ModelProvider) *Decompressor {
	t.Helper()
	d, err := NewDecompressor(p)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestKeyframeOverheadBounded(t *testing.T) {
	m := testModel(t, 16, 18)
	// Incompressible input: keyframes must appear and stay cheap relative to
	// what is actually stored.
	src := make([]byte, 4<<20)
	rng := rand.New(rand.NewSource(21))
	rng.Read(src)
	var buf bytes.Buffer
	c, err := NewCompressor(&testProvider{m}, WithMode(ModeSession), WithBlockSize(8192),
		WithKeyframeInterval(64<<10), WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	st, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf)
	if err != nil {
		t.Fatal(err)
	}
	hi, err := ReadHeader(buf.Bytes()[:headerLen])
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader(buf.Bytes()[headerLen:])
	var kfBytes, dataBytes int64
	var kfs int
	for {
		fi, payload, err := hi.NextFrame(r)
		if err != nil {
			t.Fatal(err)
		}
		if fi.IsEnd() {
			break
		}
		switch {
		case fi.Flags&FlagKeyframe != 0:
			kfs++
			kfBytes += int64(len(payload))
		case fi.Flags&FlagDictionary != 0:
		default:
			dataBytes += int64(len(payload))
		}
	}
	if kfs == 0 {
		t.Fatal("no keyframes emitted on a 4 MiB incompressible session")
	}
	if kfBytes*8 > dataBytes {
		t.Fatalf("keyframe overhead too high: %d of %d data bytes (%d keyframes)", kfBytes, dataBytes, kfs)
	}
	t.Logf("%d keyframes, %d bytes over %d data bytes (%.1f%%), total %d",
		kfs, kfBytes, dataBytes, 100*float64(kfBytes)/float64(dataBytes), st.CompBytes)
}
