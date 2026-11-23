package codec

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/radryc/compressor/model"
)

// ---------------------------------------------------------------- fixed math

func TestSigmoidTable(t *testing.T) {
	// Monotonic and bounded.
	prev := uint32(0)
	for i := 0; i < len(squashTab); i++ {
		p := uint32(squashTab[i])
		if p < prev {
			t.Fatalf("squashTab not monotonic at %d", i)
		}
		if p < 1 || p > ProbOne-1 {
			t.Fatalf("squashTab out of range at %d: %d", i, p)
		}
		prev = p
	}
	// Accuracy against float64.
	for _, x := range []float64{-12, -8, -4, -1, 0, 1, 4, 8, 12} {
		got := float64(sigmoidQ16(int32(x*65536))) / ProbOne
		want := 1 / (1 + math.Exp(-x))
		if math.Abs(got-want) > 2e-4 {
			t.Errorf("sigmoid(%v)=%v want %v", x, got, want)
		}
	}
	if got := sigmoidQ16(0); got < 32700 || got > 32800 {
		t.Errorf("sigmoid(0)=%d", got)
	}
}

func TestExpNeg(t *testing.T) {
	for _, x := range []float64{0, 0.1, 0.5, 1, 2, 5, 11, 16} {
		// Compare against exp of the Q16-rounded input: the fixed-point
		// representation itself is the dominant error term. In the deep tail
		// (x>=8) the Q30 result only has a handful of significant bits, which
		// is harmless: those probabilities are far below the coder's 1/65536
		// floor.
		q := int64(x * 65536)
		got := float64(expNeg(q)) / (1 << 30)
		want := math.Exp(-float64(q) / 65536)
		// The exponent is stored in Q16, so exp(-x) cannot be more accurate
		// than ~1.5e-5 relative; that is exactly the resolution of the Q16
		// probability the LUT produces, so it costs nothing.
		tol := 3e-5
		if x >= 12 {
			tol = 1e-2
		}
		if math.Abs(got-want) > tol*math.Max(want, 1e-6) {
			t.Errorf("expNeg(%v)=%v want %v", x, got, want)
		}
	}
}

func TestStretchInvertsSquash(t *testing.T) {
	for i := 1; i < ProbOne; i += 97 {
		l := stretch(uint32(i))
		p := squash(l)
		if math.Abs(float64(p)-float64(i)) > 700 {
			t.Fatalf("stretch/squash drift at p=%d: logit=%d back=%d", i, l, p)
		}
	}
	if got := stretch(ProbOne / 2); got < -40 || got > 40 {
		t.Errorf("stretch(0.5)=%d", got)
	}
}

func TestTanhQ11(t *testing.T) {
	for _, x := range []float64{-6, -1, 0, 1, 6} {
		got := float64(tanhQ11(int32(x*65536))) / ActOne
		want := math.Tanh(x)
		if math.Abs(got-want) > 1e-3 {
			t.Errorf("tanh(%v)=%v want %v", x, got, want)
		}
	}
}

// ------------------------------------------------------------------ dot path

func TestDotI16(t *testing.T) {
	ref := func(a, b []int16) int32 {
		var s int32
		for i := range a {
			s += int32(a[i]) * int32(b[i])
		}
		return s
	}
	rng := rand.New(rand.NewSource(1))
	for n := 0; n <= 200; n++ {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := 0; i < n; i++ {
			a[i] = int16(rng.Intn(4097) - 2048) // Q11 activation bound
			b[i] = int16(rng.Intn(16385) - 8192)
		}
		if got, want := dotI16(a, b), ref(a, b); got != want {
			t.Fatalf("dotI16 n=%d got %d want %d", n, got, want)
		}
	}
}

func TestDotI16OverflowBound(t *testing.T) {
	// Worst case allowed by model.MaxHidden and the +-4 weight clamp.
	n := model.MaxHidden
	a := make([]int16, n)
	b := make([]int16, n)
	for i := 0; i < n; i++ {
		a[i] = ActOne
		b[i] = 8192
	}
	got := dotI16(a, b)
	if got <= 0 {
		t.Fatalf("accumulator overflowed: %d", got)
	}
	if int64(got) != int64(n)*int64(ActOne)*8192 {
		t.Fatalf("dot mismatch: %d", got)
	}
}

// -------------------------------------------------------------------- coder

func TestArithmeticCoderBits(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 20; trial++ {
		n := 1 + rng.Intn(5000)
		bits := make([]uint32, n)
		probs := make([]uint32, n)
		for i := range bits {
			probs[i] = uint32(1 + rng.Intn(ProbOne-1))
			if rng.Intn(2) == 0 {
				bits[i] = 1
			}
		}
		var e ArithmeticEncoder
		e.Reset(make([]byte, 0, n))
		for i := range bits {
			e.EncodeBit(probs[i], bits[i])
		}
		out := e.Finish()

		var d ArithmeticDecoder
		d.Reset(out)
		for i := range bits {
			if got := d.DecodeBit(probs[i]); got != bits[i] {
				t.Fatalf("trial %d bit %d: got %d want %d", trial, i, got, bits[i])
			}
		}
	}
}

func TestArithmeticCoderSkewed(t *testing.T) {
	// A near-certain prediction must stay decodable and cost almost nothing.
	var e ArithmeticEncoder
	e.Reset(nil)
	const n = 100000
	for i := 0; i < n; i++ {
		e.EncodeBit(65535, 1)
	}
	out := e.Finish()
	if len(out) > n/100 {
		t.Fatalf("skewed stream too large: %d bytes", len(out))
	}
	var d ArithmeticDecoder
	d.Reset(out)
	for i := 0; i < n; i++ {
		if d.DecodeBit(65535) != 1 {
			t.Fatalf("bit %d decoded 0", i)
		}
	}
}

// ------------------------------------------------------------------- engine

func syntheticModel(t *testing.T, h, id int, scale float32) *model.Model {
	t.Helper()
	w := model.NewWeights(h, id)
	rng := rand.New(rand.NewSource(int64(id)*31 + int64(h)))
	fill := func(s []float32) {
		for i := range s {
			s[i] = (rng.Float32()*2 - 1) * scale
		}
	}
	fill(w.InProj)
	fill(w.WHH)
	fill(w.BHH)
	fill(w.WBit)
	fill(w.BBit)
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

func roundTrip(t *testing.T, m *model.Model, cfg Config, src []byte) (int, bool) {
	t.Helper()
	enc := NewBlockEngine(m, cfg)
	out, pt := enc.CompressBlock(m, src, make([]byte, 0, len(src)+64))
	dec := NewBlockEngine(m, cfg)
	var got []byte
	if pt {
		got = out
	} else {
		got = dec.DecompressBlock(m, out, len(src), nil)
	}
	if !bytes.Equal(got, src) {
		t.Fatalf("roundtrip mismatch (len=%d pt=%v)", len(src), pt)
	}
	return len(out), pt
}

func TestEngineRoundTrip(t *testing.T) {
	m := syntheticModel(t, 16, 1, 0.3)
	cfg := DefaultConfig()
	inputs := map[string][]byte{
		"empty": {},
		"one":   {0x00},
		"nul":   make([]byte, 512),
		"ff":    bytes.Repeat([]byte{0xFF}, 300),
		"alt":   bytes.Repeat([]byte{0x00, 0xFF}, 200),
	}
	rng := rand.New(rand.NewSource(3))
	rnd := make([]byte, 8192)
	rng.Read(rnd)
	inputs["rand"] = rnd

	for name, src := range inputs {
		roundTrip(t, m, cfg, src)
		t.Logf("%s: %d bytes", name, len(src))
	}
}

func TestEngineRoundTripMultiBlock(t *testing.T) {
	m := syntheticModel(t, 8, 2, 0.2)
	cfg := ConfigForBlock(4096)
	rng := rand.New(rand.NewSource(4))
	src := make([]byte, 64<<10)
	for i := range src {
		src[i] = byte('a' + rng.Intn(20))
	}
	enc := NewBlockEngine(m, cfg)
	dec := NewBlockEngine(m, cfg)
	for off := 0; off < len(src); off += 4096 {
		end := off + 4096
		if end > len(src) {
			end = len(src)
		}
		block := src[off:end]
		out, pt := enc.CompressBlock(m, block, make([]byte, 0, len(block)+64))
		var got []byte
		if pt {
			got = out
		} else {
			got = dec.DecompressBlock(m, out, len(block), nil)
		}
		if !bytes.Equal(got, block) {
			t.Fatalf("block at %d mismatch", off)
		}
	}
}

func TestPrimeMatchesCoding(t *testing.T) {
	// Priming with raw bytes must leave exactly the state that coding them
	// leaves; this is what makes session-mode passthrough frames decodable.
	m := syntheticModel(t, 8, 3, 0.25)
	cfg := DefaultConfig()
	data := []byte("the quick brown fox jumps over the lazy dog, the quick brown fox")
	tail := []byte("the quick brown fox again")

	a := NewBlockEngine(m, cfg)
	a.Prime(data)
	outA := a.CompressFrame(tail, nil)

	b := NewBlockEngine(m, cfg)
	// Code the same bytes through a throwaway coder, then keep coding.
	_ = b.CompressFrame(data, nil)
	outB := b.CompressFrame(tail, nil)

	if !bytes.Equal(outA, outB) {
		t.Fatalf("prime and coding diverged:\n %x\n %x", outA, outB)
	}
	// And it must decode from the primed side.
	c := NewBlockEngine(m, cfg)
	c.Prime(data)
	if got := c.DecompressFrame(outA, len(tail), nil); !bytes.Equal(got, tail) {
		t.Fatalf("primed decode mismatch: %q", got)
	}
}

func TestSnapshotRestore(t *testing.T) {
	m := syntheticModel(t, 8, 4, 0.25)
	cfg := DefaultConfig()
	head := []byte("0123456789abcdef0123456789abcdef")
	tail := []byte("0123456789abcdefXYZ")

	enc := NewBlockEngine(m, cfg)
	enc.Prime(head)
	snap := enc.Snapshot(nil)
	comp := enc.CompressFrame(tail, nil)

	dec := NewBlockEngine(m, cfg)
	if err := dec.Restore(m, snap); err != nil {
		t.Fatal(err)
	}
	if got := dec.DecompressFrame(comp, len(tail), nil); !bytes.Equal(got, tail) {
		t.Fatalf("resumed decode mismatch: %q", got)
	}
}

func TestBlendModes(t *testing.T) {
	m := syntheticModel(t, 8, 5, 0.3)
	src := bytes.Repeat([]byte("GET /api/v1/orders HTTP/1.1\n"), 200)
	for name, b := range map[string]Blend{
		"full":   {GRU: true, Match: true, Table: true, Run: true},
		"gru":    {GRU: true},
		"match":  {Match: true},
		"table":  {Table: true},
		"run":    {Run: true},
		"none":   {},
		"gruTbl": {GRU: true, Table: true},
	} {
		cfg := DefaultConfig()
		cfg.Blend = b
		n, _ := roundTrip(t, m, cfg, src)
		t.Logf("%-7s %6d -> %6d bytes", name, len(src), n)
	}
}
