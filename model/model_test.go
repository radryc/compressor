package model

import (
	"bytes"
	"context"
	"math"
	"sync"
	"testing"
)

func weights(t *testing.T, h, id int) *Weights {
	t.Helper()
	w := NewWeights(h, id)
	for i := range w.InProj {
		w.InProj[i] = float32(math.Sin(float64(i))) * 3
	}
	for i := range w.WHH {
		w.WHH[i] = float32(math.Cos(float64(i))) * 20
	}
	for i := range w.WBit {
		w.WBit[i] = float32(math.Sin(float64(i)*7)) * 20
	}
	for i := range w.BHH {
		w.BHH[i] = float32(i) / 10
	}
	for i := range w.BBit {
		w.BBit[i] = float32(i)
	}
	return w
}

func TestEncodeParse(t *testing.T) {
	for _, h := range []int{1, 8, 32, MaxHidden} {
		w := weights(t, h, 42)
		raw, err := w.Encode()
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) != 16+NumFloats(h)*4 {
			t.Fatalf("artifact length %d", len(raw))
		}
		m, err := Parse(raw)
		if err != nil {
			t.Fatal(err)
		}
		if m.ID != 42 || m.HiddenDim != h {
			t.Fatalf("header mismatch")
		}
		if len(m.InProj) != 256*3*h || len(m.WHH) != 3*h*h || len(m.WBit) != 8*h {
			t.Fatal("slice sizes")
		}
		// Weights are clamped to +-4 (Q11 +-8192).
		for _, v := range m.WHH {
			if v > maxWeightQ11 || v < -maxWeightQ11 {
				t.Fatalf("weight not clamped: %d", v)
			}
		}
		if m.Hash == 0 {
			t.Fatal("hash not set")
		}
	}
}

func TestParseRejectsBadArtifacts(t *testing.T) {
	w := weights(t, 8, 1)
	good, _ := w.Encode()

	cases := map[string]func([]byte) []byte{
		"empty":       func(b []byte) []byte { return nil },
		"bad magic":   func(b []byte) []byte { b[0] = 'X'; return b },
		"bad version": func(b []byte) []byte { b[4] = 99; return b },
		"short":       func(b []byte) []byte { return b[:len(b)-1] },
		"long":        func(b []byte) []byte { return append(b, 0) },
		"zero hidden": func(b []byte) []byte { b[6], b[7] = 0, 0; return b },
	}
	for name, mutate := range cases {
		b := mutate(append([]byte(nil), good...))
		if _, err := Parse(b); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

type memLoader struct {
	raw   []byte
	calls int
	mu    sync.Mutex
}

func (l *memLoader) Load(context.Context, uint32) ([]byte, error) {
	l.mu.Lock()
	l.calls++
	l.mu.Unlock()
	return l.raw, nil
}

func (l *memLoader) LoadHash(context.Context, uint32, uint32) ([]byte, error) {
	return l.Load(context.Background(), 0)
}

func TestRegistryGetByHash(t *testing.T) {
	w := weights(t, 8, 7)
	raw, _ := w.Encode()
	good, _ := Parse(raw)
	ld := &memLoader{raw: raw}
	reg := NewRegistry(ld)

	got, err := reg.Get(context.Background(), 7, good.Hash)
	if err != nil {
		t.Fatal(err)
	}
	if got.Hash != good.Hash {
		t.Fatal("hash mismatch")
	}
	// A different hash must be rejected, never silently substituted.
	if _, err := reg.Get(context.Background(), 7, good.Hash^0xFFFF); err == nil {
		t.Fatal("hash mismatch accepted")
	}
}

func TestRegistrySingleFlight(t *testing.T) {
	raw, _ := weights(t, 8, 3).Encode()
	m, _ := Parse(raw)
	ld := &memLoader{raw: raw}
	reg := NewRegistry(ld)

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := reg.Get(context.Background(), 3, m.Hash); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if ld.calls != 1 {
		t.Fatalf("loader called %d times, want 1", ld.calls)
	}
}

func TestHotSwapConcurrent(t *testing.T) {
	reg := NewRegistry(nil)
	models := make([]*Model, 16)
	for i := range models {
		raw, _ := weights(t, 8, 100+i).Encode()
		m, _ := Parse(raw)
		models[i] = m
	}
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			reg.HotSwap("d", models[i%len(models)])
			if _, ok := reg.Active("d"); !ok {
				t.Error("active model disappeared")
			}
		}(i)
	}
	wg.Wait()
	m, ok := reg.Active("d")
	if !ok {
		t.Fatal("no active model")
	}
	// Whatever won, it must be one of the registered artifacts.
	found := false
	for _, want := range models {
		if m.Hash == want.Hash {
			found = true
		}
	}
	if !found {
		t.Fatal("active model is not one that was swapped in")
	}
}

func TestParseDeterministicAcrossLoads(t *testing.T) {
	raw, _ := weights(t, 16, 5).Encode()
	a, _ := Parse(raw)
	b, _ := Parse(bytes.Clone(raw))
	if a.Hash != b.Hash {
		t.Fatal("hash not stable")
	}
	for i := range a.WHH {
		if a.WHH[i] != b.WHH[i] {
			t.Fatalf("quantisation not deterministic at %d", i)
		}
	}
}
