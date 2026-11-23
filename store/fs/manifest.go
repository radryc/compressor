package fs

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/radryc/compressor/model"
)

// Active records which artifact is currently promoted for a dataset. It is the
// durable form of Registry.HotSwap: the compressor resolves the dataset to this
// model id, and decompression is unaffected because archives carry their own
// (id, hash) pair.
type Active struct {
	Dataset string    `json:"dataset"`
	ModelID uint32    `json:"model_id"`
	Hash    uint32    `json:"hash"`
	EvalBPB float64   `json:"eval_bpb"`
	Created time.Time `json:"created"`
}

func sanitize(dataset string) string {
	var b strings.Builder
	for _, r := range dataset {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "_"
	}
	return b.String()
}

func (s *Store) activePath(dataset string) string {
	return filepath.Join(s.dir, "active-"+sanitize(dataset)+".json")
}

// SetActive promotes a model for a dataset.
func (s *Store) SetActive(a Active) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	if a.Created.IsZero() {
		a.Created = time.Now().UTC()
	}
	raw, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.activePath(a.Dataset), append(raw, '\n'), 0o644)
}

// GetActive returns the promoted model for a dataset.
func (s *Store) GetActive(dataset string) (Active, error) {
	var a Active
	raw, err := os.ReadFile(s.activePath(dataset))
	if err != nil {
		return a, err
	}
	if err := json.Unmarshal(raw, &a); err != nil {
		return a, fmt.Errorf("parse active manifest for %q: %w", dataset, err)
	}
	return a, nil
}

// ListActive returns every promotion manifest in the directory.
func (s *Store) ListActive() ([]Active, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var out []Active
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, "active-") || !strings.HasSuffix(name, ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.dir, name))
		if err != nil {
			continue
		}
		var a Active
		if err := json.Unmarshal(raw, &a); err == nil {
			out = append(out, a)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Dataset < out[j].Dataset })
	return out, nil
}

// Artifact describes one stored model file.
type Artifact struct {
	ID      uint32
	Hidden  int
	Hash    uint32
	Dataset string
	Created time.Time
	Bytes   int64
}

// List returns every id-addressed artifact, annotated with the dataset that
// promoted it (if any).
func (s *Store) List() ([]Artifact, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	active, _ := s.ListActive()
	byID := make(map[uint32]string, len(active))
	for _, a := range active {
		byID[a.ModelID] = a.Dataset
	}
	var out []Artifact
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".bin") {
			continue
		}
		// Content-addressed copies are prefixed with "h-" and are duplicates
		// of an id-addressed artifact.
		if strings.HasPrefix(name, "h-") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.dir, name))
		if err != nil {
			continue
		}
		m, err := model.Parse(raw)
		if err != nil {
			continue
		}
		info, _ := e.Info()
		var created time.Time
		if info != nil {
			created = info.ModTime()
		}
		out = append(out, Artifact{
			ID: m.ID, Hidden: m.HiddenDim, Hash: m.Hash,
			Dataset: byID[m.ID], Created: created, Bytes: int64(len(raw)),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// SingleModelID returns the only id-addressed artifact id in the directory.
func (s *Store) SingleModelID() (uint32, error) {
	arts, err := s.List()
	if err != nil {
		return 0, err
	}
	ids := map[uint32]bool{}
	for _, a := range arts {
		ids[a.ID] = true
	}
	if len(ids) != 1 {
		return 0, fmt.Errorf("expected exactly one model, found %d", len(ids))
	}
	for id := range ids {
		return id, nil
	}
	return 0, fmt.Errorf("no models")
}
