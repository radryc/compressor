package fs

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/radryc/compressor/model"
)

func artifact(t *testing.T, id int) []byte {
	t.Helper()
	w := model.NewWeights(8, id)
	for i := range w.InProj {
		w.InProj[i] = float32(i%7) / 10
	}
	for i := range w.WBit {
		w.WBit[i] = float32(i%5) / 10
	}
	raw, err := w.Encode()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestPutAndLoadByHash(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	raw := artifact(t, 5)
	m, err := s.Put(raw)
	if err != nil {
		t.Fatal(err)
	}
	// Content-addressed lookup.
	got, err := s.LoadHash(context.Background(), m.ID, m.Hash)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(raw) {
		t.Fatal("content-addressed load mismatch")
	}
	// A mismatched hash must error rather than return the wrong weights.
	other := artifact(t, 6)
	om, _ := model.Parse(other)
	if _, err := s.LoadHash(context.Background(), m.ID, om.Hash); err == nil {
		t.Fatal("hash mismatch not detected")
	}
}

func TestActiveManifest(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	raw := artifact(t, 9)
	m, _ := s.Put(raw)
	if err := s.SetActive(Active{Dataset: "otel/traces", ModelID: m.ID, Hash: m.Hash, EvalBPB: 0.42, Created: time.Now()}); err != nil {
		t.Fatal(err)
	}
	a, err := s.GetActive("otel/traces")
	if err != nil {
		t.Fatal(err)
	}
	if a.ModelID != m.ID || a.Hash != m.Hash {
		t.Fatalf("bad manifest: %+v", a)
	}
	list, err := s.ListActive()
	if err != nil || len(list) != 1 {
		t.Fatalf("list: %v %v", list, err)
	}
}

func TestListAndSingleModel(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	raw := artifact(t, 3)
	if _, err := s.Put(raw); err != nil {
		t.Fatal(err)
	}
	arts, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(arts) != 1 || arts[0].ID != 3 || arts[0].Hidden != 8 {
		t.Fatalf("bad list: %+v", arts)
	}
	id, err := s.SingleModelID()
	if err != nil || id != 3 {
		t.Fatalf("single: %d %v", id, err)
	}
	// A second model makes "single" ambiguous.
	if _, err := s.Put(artifact(t, 4)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SingleModelID(); err == nil {
		t.Fatal("expected ambiguity error")
	}
}

func TestLoadAllSkipsGarbage(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	if _, err := s.Put(artifact(t, 1)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "9.bin"), []byte("garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := model.NewRegistry(s)
	if err := s.LoadAll(reg); err != nil {
		t.Fatal(err)
	}
	if reg.Count() != 1 {
		t.Fatalf("registry has %d models", reg.Count())
	}
}
