package stream

import (
	"bytes"
	"context"
	"testing"

	"github.com/radryc/compressor/model"
)

// FuzzDecompress feeds arbitrary bytes to the decompressor. It must never
// panic, never allocate unboundedly, and must fail loudly on anything that is
// not a valid stream.
func FuzzDecompress(f *testing.F) {
	w := model.NewWeights(4, 1)
	raw, err := w.Encode()
	if err != nil {
		f.Fatal(err)
	}
	m, err := model.Parse(raw)
	if err != nil {
		f.Fatal(err)
	}
	prov := &testProvider{m: m}

	// Seed with a real stream so the fuzzer starts from valid input.
	src := corpus(4096)
	var buf bytes.Buffer
	c, _ := NewCompressor(prov, WithBlockSize(1024), WithWorkers(1))
	if _, err := c.Compress(context.Background(), "d", bytes.NewReader(src), &buf); err == nil {
		f.Add(buf.Bytes())
	}
	f.Add([]byte("NGR3"))
	f.Add(bytes.Repeat([]byte{0xFF}, 64))

	f.Fuzz(func(t *testing.T, data []byte) {
		d, err := NewDecompressor(prov, WithWorkers(1))
		if err != nil {
			t.Fatal(err)
		}
		var out bytes.Buffer
		// Errors are expected; panics and unbounded memory are not.
		_, _ = d.Decompress(context.Background(), bytes.NewReader(data), &out)
	})
}
