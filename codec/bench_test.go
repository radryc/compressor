package codec

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

func benchCorpus(n int) []byte {
	// Mix of structured, semi-structured and random bytes, similar in spirit
	// to a batch of OTel spans.
	line := []byte(`{"traceId":"4bf92f3577b34da6a3ce929d0e0e4736","spanId":"00f067aa0ba902b7","name":"POST /api/v1/orders","kind":2,"startTimeUnixNano":"1732347421104829000","status":{"code":1}}` + "\n")
	out := make([]byte, 0, n)
	rng := rand.New(rand.NewSource(9))
	for len(out) < n {
		c := append([]byte(nil), line...)
		for i := 0; i < 4; i++ {
			p := 12 + rng.Intn(90)
			if p < len(c) && c[p] >= '0' && c[p] <= '9' {
				c[p] = byte('0' + rng.Intn(10))
			}
		}
		out = append(out, c...)
	}
	return out[:n]
}

func BenchmarkCompress(b *testing.B) {
	src := benchCorpus(1 << 20)
	for _, h := range []int{16, 32, 64} {
		m := syntheticModel(&testing.T{}, h, 1, 0.2)
		eng := NewBlockEngine(m, DefaultConfig())
		b.Run(fmt.Sprintf("H%d", h), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			dst := make([]byte, 0, len(src))
			for i := 0; i < b.N; i++ {
				eng.CompressBlock(m, src, dst)
			}
		})
	}
}

func BenchmarkDecompress(b *testing.B) {
	src := benchCorpus(1 << 20)
	for _, h := range []int{16, 32, 64} {
		m := syntheticModel(&testing.T{}, h, 2, 0.2)
		eng := NewBlockEngine(m, DefaultConfig())
		comp, pt := eng.CompressBlock(m, src, make([]byte, 0, len(src)))
		if pt {
			b.Fatalf("expected compression")
		}
		b.Run(fmt.Sprintf("H%d", h), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			dec := NewBlockEngine(m, DefaultConfig())
			dst := make([]byte, 0, len(src))
			for i := 0; i < b.N; i++ {
				dec.DecompressBlock(m, comp, len(src), dst)
			}
		})
	}
}

func BenchmarkMatchOnly(b *testing.B) {
	src := benchCorpus(1 << 20)
	m := syntheticModel(&testing.T{}, 16, 3, 0.2)
	cfg := DefaultConfig()
	cfg.Blend = Blend{Match: true}
	eng := NewBlockEngine(m, cfg)
	b.SetBytes(int64(len(src)))
	dst := make([]byte, 0, len(src))
	for i := 0; i < b.N; i++ {
		eng.CompressBlock(m, src, dst)
	}
}

func TestRatioOnStructuredData(t *testing.T) {
	src := benchCorpus(1 << 20)
	for _, h := range []int{16, 32} {
		m := syntheticModel(t, h, 1, 0.2)
		var buf bytes.Buffer
		for _, cfg := range []struct {
			name  string
			blend Blend
		}{
			{"full", Blend{GRU: true, Match: true, Table: true, Run: true}},
			{"match", Blend{Match: true}},
			{"gru+table", Blend{GRU: true, Table: true}},
		} {
			c := DefaultConfig()
			c.Blend = cfg.blend
			eng := NewBlockEngine(m, c)
			out, pt := eng.CompressBlock(m, src, make([]byte, 0, len(src)))
			if pt {
				out = src
			}
			buf.Reset()
			t.Logf("H=%-2d %-10s %7d -> %7d (%.3f)", h, cfg.name, len(src), len(out), float64(len(out))/float64(len(src)))
		}
	}
}
