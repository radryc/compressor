package codec

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestDeltaRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 5, 100, 4096, 70000} {
		for _, stride := range []int{0, 1, 3, 4, 768, 3072, 100000} {
			src := make([]byte, n)
			rng.Read(src)
			work := append([]byte(nil), src...)
			DeltaEncode(work, stride)
			DeltaDecode(work, stride)
			if !bytes.Equal(work, src) {
				t.Fatalf("n=%d stride=%d: delta not reversible", n, stride)
			}
		}
	}
}

func TestDetectStride(t *testing.T) {
	// A synthetic raster: rows are similar but not identical.
	rng := rand.New(rand.NewSource(2))
	for _, stride := range []int{12, 96, 768, 1536, 3072} {
		rows := 800
		b := make([]byte, stride*rows)
		base := make([]byte, stride)
		rng.Read(base)
		for r := 0; r < rows; r++ {
			copy(b[r*stride:], base)
			for i := 0; i < stride; i += 17 {
				b[r*stride+i] = byte(r * 3) // slow drift down the rows
			}
		}
		got := DetectStride(b)
		if got != stride {
			t.Errorf("raster stride %d detected as %d", stride, got)
		}
	}
	// Noise must not look like a raster.
	noise := make([]byte, 1<<16)
	rand.New(rand.NewSource(3)).Read(noise)
	if got := DetectStride(noise); got != 0 {
		t.Errorf("noise detected as raster stride %d", got)
	}
}

func TestProfileConfig(t *testing.T) {
	// The text profile must allocate the deeper context; telemetry must not.
	m := syntheticModel(t, 8, 1, 0.2)
	text := NewBlockEngine(m, ConfigForProfile(ProfileText, 1<<20))
	if text.o3 == nil || text.o4 == nil || text.word == nil {
		t.Fatal("text profile missing deeper contexts")
	}
	tel := NewBlockEngine(m, ConfigForProfile(ProfileTelemetry, 1<<20))
	if tel.o3 != nil || tel.o4 != nil || tel.word != nil {
		t.Fatal("telemetry profile allocated deeper contexts")
	}
	if text.Profile() != ProfileText || tel.Profile() != ProfileTelemetry {
		t.Fatal("profile not reported")
	}
}

func TestTextProfileRoundTrip(t *testing.T) {
	m := syntheticModel(t, 8, 2, 0.25)
	rng := rand.New(rand.NewSource(4))
	src := make([]byte, 200000)
	for i := range src {
		// Prose-like: letters and spaces.
		if rng.Intn(6) == 0 {
			src[i] = ' '
		} else {
			src[i] = byte('a' + rng.Intn(26))
		}
	}
	for _, p := range []Profile{ProfileTelemetry, ProfileText, ProfileImage} {
		cfg := ConfigForProfile(p, 1<<16)
		enc := NewBlockEngine(m, cfg)
		out, pt := enc.CompressBlock(m, src, make([]byte, 0, len(src)))
		dec := NewBlockEngine(m, cfg)
		var got []byte
		if pt {
			got = out
		} else {
			got = dec.DecompressBlock(m, out, len(src), nil)
		}
		if !bytes.Equal(got, src) {
			t.Fatalf("profile %v: roundtrip failed", p)
		}
		t.Logf("profile %-9s -> %d bytes", p, len(out))
	}
}

func TestPaethRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, n := range []int{0, 1, 5, 100, 4096, 70000} {
		for _, stride := range []int{0, 1, 3, 4, 768, 100000} {
			src := make([]byte, n)
			rng.Read(src)
			work := append([]byte(nil), src...)
			PaethEncode(work, stride)
			PaethDecode(work, stride)
			if !bytes.Equal(work, src) {
				t.Fatalf("n=%d stride=%d: paeth not reversible", n, stride)
			}
		}
	}
}

func TestChooseTransform(t *testing.T) {
	// A raster: a fixed row pattern plus a slow vertical drift. Only the true
	// row stride makes the vertical residual constant.
	const w = 384
	base := make([]byte, w)
	rng := rand.New(rand.NewSource(12))
	acc := 0
	for c := 0; c < w; c++ {
		acc += rng.Intn(5) - 2
		base[c] = byte(acc)
	}
	linear := make([]byte, w*200)
	for r := 0; r < 200; r++ {
		for c := 0; c < w; c++ {
			linear[r*w+c] = byte(int(base[c]) + r*3)
		}
	}
	tr, stride := ChooseTransform(linear)
	if tr == TransformNone {
		t.Fatal("no transform chosen for a linear raster")
	}
	if stride != w {
		t.Fatalf("stride %d, want %d", stride, w)
	}

	// Noise: no transform must be chosen.
	noise := make([]byte, 1<<16)
	rand.New(rand.NewSource(11)).Read(noise)
	if tr, _ := ChooseTransform(noise); tr != TransformNone {
		t.Fatalf("transform %v chosen for noise", tr)
	}

	// Whichever is chosen must beat the untransformed entropy.
	for _, b := range [][]byte{linear} {
		base := entropyBits(b, 1<<16)
		w := append([]byte(nil), b...)
		tr.Apply(w, stride)
		if got := entropyBits(w, 1<<16); got >= base {
			t.Fatalf("chosen transform did not reduce entropy: %f vs %f", got, base)
		}
	}
}

// The text profile must beat telemetry on prose. This guards the deeper
// contexts: an earlier version indexed them once per byte instead of once per
// bit-tree node, which made the profile a net loss.
func TestTextProfileBeatsTelemetryOnProse(t *testing.T) {
	m := syntheticModel(t, 16, 30, 0.25)
	words := []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
		"and", "then", "walks", "away", "while", "nobody", "watches", "carefully"}
	var prose []byte
	rng := rand.New(rand.NewSource(31))
	for len(prose) < 400000 {
		prose = append(prose, words[rng.Intn(len(words))]...)
		if rng.Intn(9) == 0 {
			prose = append(prose, '.', '\n')
		} else {
			prose = append(prose, ' ')
		}
	}
	size := func(p Profile) int {
		cfg := ConfigForProfile(p, 1<<16)
		e := NewBlockEngine(m, cfg)
		out, pt := e.CompressBlock(m, prose, make([]byte, 0, len(prose)))
		if pt {
			return len(prose)
		}
		return len(out)
	}
	tel, txt := size(ProfileTelemetry), size(ProfileText)
	t.Logf("telemetry=%d text=%d", tel, txt)
	if txt >= tel {
		t.Fatalf("text profile (%d) did not beat telemetry (%d)", txt, tel)
	}
}
