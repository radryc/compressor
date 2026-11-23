package codec

import (
	"github.com/radryc/compressor/model"
)

// GRUState is a fully quantised, forward-only GRU.
//
// All arithmetic is integer: weights are Q11, activations are Q11 and partial
// sums are Q22 accumulated in int32. The accumulator bound is
// H * |w|max * |h|max = 96 * 8192 * 2048 < 2^31, enforced by model.MaxHidden
// and by clamping weight quantisation to +-4.
type GRUState struct {
	m *model.Model
	h []int16 // Q11 hidden state
	r []int16 // Q11 reset gate
	z []int16 // Q11 update gate
	n []int16 // Q11 candidate

	// bitLogit[b] is the Q7 natural-log logit for bit b (MSB first) of the
	// next byte.
	bitLogit [8]int32
}

func (s *GRUState) Reset(m *model.Model) {
	s.m = m
	H := m.HiddenDim
	if cap(s.h) < H {
		s.h = make([]int16, H)
		s.r = make([]int16, H)
		s.z = make([]int16, H)
		s.n = make([]int16, H)
	} else {
		s.h = s.h[:H]
		s.r = s.r[:H]
		s.z = s.z[:H]
		s.n = s.n[:H]
		clear(s.h)
		clear(s.r)
		clear(s.z)
		clear(s.n)
	}
	s.bitLogit = [8]int32{}
}

// Step advances the recurrence by one input byte and recomputes the eight bit
// logits for the next byte.
func (s *GRUState) Step(prevByte int) {
	m := s.m
	H := m.HiddenDim
	inOff := prevByte * 3 * H
	h := s.h

	for i := 0; i < H; i++ {
		hr := dotI16(m.WHH[i*H:i*H+H], h)
		hz := dotI16(m.WHH[(H+i)*H:(H+i)*H+H], h)
		hn := dotI16(m.WHH[(2*H+i)*H:(2*H+i)*H+H], h)

		sr := int64(hr>>AccShift) + int64(m.InProj[inOff+i]) + int64(m.BHH[i])
		sz := int64(hz>>AccShift) + int64(m.InProj[inOff+H+i]) + int64(m.BHH[H+i])

		ri := sigmoidQ11(clampLogitInput(sr))
		zi := sigmoidQ11(clampLogitInput(sz))
		// candidate: tanh(in_n + r * (W_hn h + b_hn))
		hn16 := int32(hn >> AccShift)
		pre := int64(m.InProj[inOff+2*H+i]) + int64((int64(ri)*int64(hn16))>>ActShift)

		s.r[i] = ri
		s.z[i] = zi
		s.n[i] = tanhQ11(clampLogitInput(pre))
	}

	for i := 0; i < H; i++ {
		zi := int32(s.z[i])
		hi := int32(s.h[i])
		ni := int32(s.n[i])
		s.h[i] = int16(((ActOne-zi)*ni + zi*hi) >> ActShift)
	}

	for b := 0; b < 8; b++ {
		acc := dotI16(m.WBit[b*H:b*H+H], s.h)
		s.bitLogit[b] = clampLogit(int32(acc>>15) + m.BBit[b])
	}
}

// clampLogitInput bounds a Q16 pre-activation to +-16 nats before the
// sigmoid/tanh saturating lookup.
func clampLogitInput(v int64) int32 {
	const lim = int64(16 * ProbOne)
	if v > lim {
		return int32(lim)
	}
	if v < -lim {
		return int32(-lim)
	}
	return int32(v)
}

// clampLogit bounds a Q7 logit to the LUT range.
func clampLogit(v int32) int32 {
	if v < -LogitLimit {
		return -LogitLimit
	}
	if v > LogitLimit-1 {
		return LogitLimit - 1
	}
	return v
}
