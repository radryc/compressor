package train

import (
	"bytes"
	"math/rand"
	"testing"
	"time"

	"github.com/radryc/compressor/codec"
	"github.com/radryc/compressor/model"
)

func sampleCorpus(n int) []byte {
	line := []byte(`{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"checkout-api"}}]},"scopeSpans":[{"spans":[{"traceId":"4bf92f3577b34da6a3ce929d0e0e4736","spanId":"00f067aa0ba902b7","name":"POST /api/v1/orders","kind":2,"startTimeUnixNano":"1732347421104829000","status":{"code":1}}]}]}]}` + "\n")
	out := make([]byte, 0, n)
	rng := rand.New(rand.NewSource(3))
	for len(out) < n {
		c := append([]byte(nil), line...)
		for i := 0; i < 5; i++ {
			p := 20 + rng.Intn(140)
			if p < len(c) && c[p] >= '0' && c[p] <= '9' {
				c[p] = byte('0' + rng.Intn(10))
			}
		}
		out = append(out, c...)
	}
	return out[:n]
}

func TestTrainReducesLoss(t *testing.T) {
	data := sampleCorpus(120000)
	cfg := Config{Hidden: 8, SeqLen: 32, Epochs: 2, LR: 5e-3, Seed: 1, Holdout: 0.1}
	cfg.Logf = t.Logf
	start := time.Now()
	res, w, err := Train(data, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("trained in %v: %v", time.Since(start), res.TrainBPB)
	if len(res.TrainBPB) < 2 {
		t.Fatal("expected two epochs")
	}
	if res.TrainBPB[len(res.TrainBPB)-1] >= res.TrainBPB[0] {
		t.Fatalf("loss did not decrease: %v", res.TrainBPB)
	}
	if res.TrainBPB[len(res.TrainBPB)-1] > 6 {
		t.Fatalf("loss too high: %v", res.TrainBPB)
	}

	raw, err := w.Encode()
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	// The quantised codec must reproduce the trained behaviour closely enough
	// to beat the float-model BPB by a wide margin over raw bytes.
	eng := codec.NewBlockEngine(m, codec.DefaultConfig())
	src := sampleCorpus(20000)
	out, pt := eng.CompressBlock(m, src, make([]byte, 0, len(src)))
	if pt {
		t.Fatalf("trained model failed to compress (out=%d)", len(out))
	}
	ratio := float64(len(out)) / float64(len(src))
	t.Logf("codec ratio with trained quantised model: %.4f (%.2f bpb)", ratio, ratio*8)
	if ratio > 0.5 {
		t.Fatalf("ratio %.3f too poor", ratio)
	}
	dec := codec.NewBlockEngine(m, codec.DefaultConfig())
	if got := dec.DecompressBlock(m, out, len(src), nil); !bytes.Equal(got, src) {
		t.Fatal("roundtrip failed with trained model")
	}
}
