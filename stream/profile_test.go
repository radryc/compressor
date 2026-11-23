package stream

import (
	"bytes"
	"context"
	"math/rand"
	"testing"

	"github.com/radryc/compressor/codec"
)

func TestClassify(t *testing.T) {
	rng := rand.New(rand.NewSource(5))

	json := bytes.Repeat([]byte(`{"traceId":"4bf92f3577b34da6a3ce929d0e0e4736","spanId":"00f067aa0ba902b7","kind":2}`+"\n"), 200)
	prose := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog and then walks away slowly\n"), 200)
	noise := make([]byte, 1<<15)
	rng.Read(noise)

	// A raster with a 384-byte row stride.
	raster := make([]byte, 384*80)
	for r := 0; r < 80; r++ {
		for c := 0; c < 384; c++ {
			raster[r*384+c] = byte((c/3 + r) % 251)
		}
	}

	cases := []struct {
		name string
		b    []byte
		want codec.Profile
	}{
		{"json", json, codec.ProfileTelemetry},
		{"prose", prose, codec.ProfileText},
		{"noise", noise, codec.ProfileTelemetry},
		{"raster", raster, codec.ProfileImage},
		{"empty", nil, codec.ProfileTelemetry},
	}
	for _, c := range cases {
		d := Classify(c.b)
		if d.Profile != c.want {
			t.Errorf("%s: profile %v, want %v", c.name, d.Profile, c.want)
		}
		if c.want == codec.ProfileImage && d.Stride == 0 {
			t.Errorf("%s: image profile without a stride", c.name)
		}
	}
}

// A stream whose blocks have different profiles must round-trip.
func TestMixedProfilesRoundTrip(t *testing.T) {
	m := testModel(t, 16, 20)
	rng := rand.New(rand.NewSource(6))

	prose := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog, again and again\n"), 900)
	json := corpus(30000)
	raster := make([]byte, 384*400)
	for r := 0; r < 400; r++ {
		for c := 0; c < 384; c++ {
			raster[r*384+c] = byte((c/3 + r*2) % 251)
		}
	}
	noise := make([]byte, 20000)
	rng.Read(noise)

	src := append(append(append(append([]byte{}, prose...), json...), raster...), noise...)

	for _, mode := range []Mode{ModeBlocks, ModeSession} {
		st := roundTrip(t, m, src, []Option{WithMode(mode), WithBlockSize(16384), WithWorkers(3)}, nil)
		t.Logf("mode=%v %d -> %d bytes (%.3f)", mode, len(src), st.CompBytes, st.Ratio())
	}
}

func TestImageTransformHelps(t *testing.T) {
	m := testModel(t, 16, 21)
	raster := make([]byte, 768*250)
	for r := 0; r < 250; r++ {
		for c := 0; c < 768; c++ {
			raster[r*768+c] = byte((c / 3) % 256)
		}
	}
	// With the transform.
	st := roundTrip(t, m, raster, []Option{WithBlockSize(1 << 16)}, nil)
	// Without: force telemetry by compressing a raster that the classifier
	// cannot see as one (shuffle the rows).
	shuffled := make([]byte, len(raster))
	copy(shuffled, raster)
	rng := rand.New(rand.NewSource(7))
	for i := len(shuffled) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	}
	stPlain := roundTrip(t, m, shuffled, []Option{WithBlockSize(1 << 16)}, nil)
	t.Logf("raster: %d -> %d (transform) | shuffled: %d -> %d",
		len(raster), st.CompBytes, len(shuffled), stPlain.CompBytes)
	if st.CompBytes >= stPlain.CompBytes {
		t.Fatalf("transform did not help on a raster: %d vs %d", st.CompBytes, stPlain.CompBytes)
	}
}

func TestProfileStats(t *testing.T) {
	m := testModel(t, 16, 22)
	src := corpus(100000)
	c, _ := NewCompressor(&testProvider{m}, WithBlockSize(8192), WithWorkers(2))
	var buf bytes.Buffer
	st, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if st.Frames != int64((len(src)+8191)/8192) {
		t.Fatalf("frames %d", st.Frames)
	}
}
