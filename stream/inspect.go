package stream

import (
	"fmt"
	"io"

	"github.com/radryc/compressor/codec"
)

// HeaderInfo is the parsed, read-only view of an NGR3 header.
type HeaderInfo struct {
	Version   uint8
	ModelID   uint32
	ModelHash uint32
	BlockSize uint32
	FrameMax  uint32
	DictHash  uint32
	Session   bool
	Blend     codec.Blend
	h         *header
}

// ReadHeader parses a 32-byte NGR3 header without resolving a model.
func ReadHeader(b []byte) (*HeaderInfo, error) {
	h, err := unmarshalHeader(b)
	if err != nil {
		return nil, err
	}
	return &HeaderInfo{
		Version:   Version,
		ModelID:   h.modelID,
		ModelHash: h.modelHash,
		BlockSize: h.blockSize,
		FrameMax:  h.frameMax,
		DictHash:  h.dictHash,
		Session:   h.session(),
		Blend:     h.blend(),
		h:         h,
	}, nil
}

// FrameInfo describes one frame.
type FrameInfo struct {
	Seq       uint64
	RawLen    int
	CompLen   int
	Flags     uint8
	Profile   codec.Profile
	Transform codec.Transform
	Stride    int
}

// IsEnd reports whether this is the terminating frame.
func (f *FrameInfo) IsEnd() bool { return f.Flags&FlagEnd != 0 }

// NextFrame reads and validates the next frame, returning its payload.
func (h *HeaderInfo) NextFrame(r io.Reader) (*FrameInfo, []byte, error) {
	fh, payload, err := readFrame(r, h.h)
	if err != nil {
		return nil, nil, err
	}
	return &FrameInfo{
		Seq:       fh.seq,
		RawLen:    int(fh.rawLen),
		CompLen:   int(fh.compLen),
		Flags:     fh.flags,
		Profile:   codec.Profile(fh.profile),
		Transform: codec.Transform(fh.transform),
		Stride:    int(fh.stride),
	}, payload, nil
}

// Describe renders a frame's flags for humans.
func (f *FrameInfo) Describe() string {
	out := ""
	add := func(s string) {
		if out != "" {
			out += ","
		}
		out += s
	}
	if f.Flags&FlagPassthrough != 0 {
		add("passthrough")
	}
	if f.Flags&FlagKeyframe != 0 {
		add("keyframe")
	}
	if f.Flags&FlagDictionary != 0 {
		add("dictionary")
	}
	if f.Flags&FlagEnd != 0 {
		add("end")
	}
	if out == "" {
		out = "coded"
	}
	return out
}

var _ = fmt.Sprintf
