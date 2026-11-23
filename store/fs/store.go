package fs

import (
	"context"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"

	"github.com/radryc/compressor/model"
)

// Store loads and stores NGRM model artifacts on the local filesystem.
//
// Artifacts are kept content-addressed (<dir>/<hash>.bin) as well as by id
// (<dir>/<id>.bin). Decompression resolves by (id, hash), so retraining model
// 101 never makes an archive that references the old weights undecodable.
type Store struct {
	dir string
}

func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

func (s *Store) idPath(id uint32) string {
	return filepath.Join(s.dir, fmt.Sprintf("%d.bin", id))
}

func (s *Store) hashPath(hash uint32) string {
	return filepath.Join(s.dir, fmt.Sprintf("h-%08x.bin", hash))
}

// Load implements model.Loader by reading <dir>/<id>.bin.
func (s *Store) Load(_ context.Context, id uint32) ([]byte, error) {
	return os.ReadFile(s.idPath(id))
}

// LoadHash implements model.HashLoader, preferring the exact artifact.
func (s *Store) LoadHash(_ context.Context, id, hash uint32) ([]byte, error) {
	if raw, err := os.ReadFile(s.hashPath(hash)); err == nil {
		return raw, nil
	}
	raw, err := os.ReadFile(s.idPath(id))
	if err != nil {
		return nil, err
	}
	if got := crc32.Checksum(raw, crc32.MakeTable(crc32.Castagnoli)); got != hash {
		return nil, fmt.Errorf("model %d content mismatch: on disk %08x, archive expects %08x", id, got, hash)
	}
	return raw, nil
}

// Put writes an artifact content-addressed and updates the id pointer.
func (s *Store) Put(raw []byte) (*model.Model, error) {
	m, err := model.Parse(raw)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(s.hashPath(m.Hash), raw, 0o644); err != nil {
		return nil, err
	}
	if err := os.WriteFile(s.idPath(m.ID), raw, 0o644); err != nil {
		return nil, err
	}
	return m, nil
}

// LoadAll preloads every .bin model in the directory into reg.
func (s *Store) LoadAll(reg *model.Registry) error {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".bin" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.dir, e.Name()))
		if err != nil {
			continue
		}
		if m, err := model.Parse(raw); err == nil {
			reg.Register(m)
		}
	}
	return nil
}
