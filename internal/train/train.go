// Package train implements a dependency-free trainer for the byte-level GRU
// used by the codec.
//
// It exists so the full loop — sample, train, evaluate with the real codec,
// publish, hot-swap — runs without PyTorch and can be exercised in CI. The
// architecture matches model.Weights exactly:
//
//	input projection   InProj[256][3H]  (one-hot previous byte, input bias folded)
//	recurrent weights  WHH[3H][H]       (gates ordered r, z, n)
//	recurrent bias     BHH[3H]
//	bit heads          WBit[8][H]
//	bit-head bias      BBit[8]
//
// The codec quantises these weights; training here is float64 to keep the
// gradients well-conditioned.
package train

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/radryc/compressor/model"
)

// Config controls a training run.
type Config struct {
	Hidden   int
	SeqLen   int
	Epochs   int
	LR       float64
	Seed     int64
	Holdout  float64 // fraction of bytes reserved for evaluation
	Logf     func(format string, args ...any)
	MaxSteps int // 0 = unlimited
	// StepsPerEpoch caps the windows drawn per epoch (0 = derived cap).
	StepsPerEpoch int
}

// Result reports training progress.
type Result struct {
	TrainBPB []float64
	EvalBPB  float64
	Steps    int
}

func (c *Config) defaults() {
	if c.Hidden <= 0 {
		c.Hidden = 32
	}
	if c.Hidden > model.MaxHidden {
		c.Hidden = model.MaxHidden
	}
	if c.SeqLen <= 0 {
		c.SeqLen = 64
	}
	if c.Epochs <= 0 {
		c.Epochs = 3
	}
	if c.LR <= 0 {
		c.LR = 3e-3
	}
	if c.Holdout <= 0 || c.Holdout >= 0.5 {
		c.Holdout = 0.05
	}
	if c.Logf == nil {
		c.Logf = func(string, ...any) {}
	}
}

type params struct {
	H      int
	InProj []float64
	WHH    []float64
	BHH    []float64
	WBit   []float64
	BBit   []float64
}

func newParams(h int, rng *rand.Rand) *params {
	p := &params{
		H:      h,
		InProj: make([]float64, 256*3*h),
		WHH:    make([]float64, 3*h*h),
		BHH:    make([]float64, 3*h),
		WBit:   make([]float64, 8*h),
		BBit:   make([]float64, 8),
	}
	scale := 1 / math.Sqrt(float64(h))
	fill := func(s []float64) {
		for i := range s {
			s[i] = (rng.Float64()*2 - 1) * scale
		}
	}
	fill(p.InProj)
	fill(p.WHH)
	fill(p.WBit)
	return p
}

// Weights converts the trained parameters into an artifact.
func (p *params) Weights(id uint32) *model.Weights {
	w := model.NewWeights(p.H, int(id))
	copyF32 := func(dst []float32, src []float64) {
		for i := range dst {
			dst[i] = float32(src[i])
		}
	}
	copyF32(w.InProj, p.InProj)
	copyF32(w.WHH, p.WHH)
	copyF32(w.BHH, p.BHH)
	copyF32(w.WBit, p.WBit)
	copyF32(w.BBit, p.BBit)
	return w
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// stepState is what BPTT needs from one forward step.
type stepState struct {
	in    byte
	z, r  []float64
	n     []float64
	hn    []float64 // W_hn . h + b_hn
	hPrev []float64
}

// forward runs the network over (inputs, targets) and returns bits per byte.
// inputs[t] is the previous byte fed to the recurrence; targets[t] is the byte
// that the bit heads must predict. This mirrors the codec exactly.
func (p *params) forward(inputs, targets []byte, states []stepState) float64 {
	H := p.H
	h := make([]float64, H)
	hPrev := make([]float64, H)
	var loss float64

	for t := range inputs {
		copy(hPrev, h)
		base := int(inputs[t]) * 3 * H
		var st *stepState
		if states != nil {
			st = &states[t]
			st.in = inputs[t]
			st.hPrev = append(st.hPrev[:0], hPrev...)
			st.z = append(st.z[:0], make([]float64, H)...)
			st.r = append(st.r[:0], make([]float64, H)...)
			st.n = append(st.n[:0], make([]float64, H)...)
			st.hn = append(st.hn[:0], make([]float64, H)...)
		}
		for i := 0; i < H; i++ {
			sr := p.InProj[base+i] + p.BHH[i]
			sz := p.InProj[base+H+i] + p.BHH[H+i]
			hn := p.BHH[2*H+i]
			for j := 0; j < H; j++ {
				sr += p.WHH[i*H+j] * hPrev[j]
				sz += p.WHH[(H+i)*H+j] * hPrev[j]
				hn += p.WHH[(2*H+i)*H+j] * hPrev[j]
			}
			r := sigmoid(sr)
			z := sigmoid(sz)
			n := math.Tanh(p.InProj[base+2*H+i] + r*hn)
			h[i] = (1-z)*n + z*hPrev[i]
			if st != nil {
				st.r[i], st.z[i], st.n[i], st.hn[i] = r, z, n, hn
			}
		}
		target := targets[t]
		for b := 0; b < 8; b++ {
			var l float64
			for j := 0; j < H; j++ {
				l += p.WBit[b*H+j] * h[j]
			}
			l += p.BBit[b]
			pr := sigmoid(l)
			y := float64((target >> uint(7-b)) & 1)
			loss += -math.Log2(pr*y + (1-pr)*(1-y) + 1e-300)
		}
	}
	if len(inputs) == 0 {
		return 0
	}
	return loss / float64(len(inputs))
}

// backward accumulates gradients for one sequence.
func (p *params) backward(inputs, targets []byte, states []stepState, g *params) {
	H := p.H
	n := len(inputs)
	if n == 0 {
		return
	}
	dh := make([]float64, H)
	dhNext := make([]float64, H)

	// Rebuild the forward hidden vectors.
	hs := make([][]float64, n+1)
	hs[0] = make([]float64, H)
	for t := 0; t < n; t++ {
		next := make([]float64, H)
		for i := 0; i < H; i++ {
			z := states[t].z[i]
			next[i] = (1-z)*states[t].n[i] + z*hs[t][i]
		}
		hs[t+1] = next
	}

	invN := 1 / float64(n)
	for t := n - 1; t >= 0; t-- {
		st := &states[t]
		copy(dh, dhNext)
		for i := range dhNext {
			dhNext[i] = 0
		}

		// bit heads over h_t
		target := targets[t]
		for b := 0; b < 8; b++ {
			var l float64
			for j := 0; j < H; j++ {
				l += p.WBit[b*H+j] * hs[t+1][j]
			}
			l += p.BBit[b]
			pr := sigmoid(l)
			y := float64((target >> uint(7-b)) & 1)
			dl := (pr - y) / math.Ln2
			g.BBit[b] += dl * invN
			for j := 0; j < H; j++ {
				g.WBit[b*H+j] += dl * hs[t+1][j] * invN
				dh[j] += dl * p.WBit[b*H+j]
			}
		}

		base := int(st.in) * 3 * H
		for i := 0; i < H; i++ {
			dz := dh[i] * (hs[t][i] - st.n[i])
			dn := dh[i] * (1 - st.z[i])
			dhNext[i] += dh[i] * st.z[i]

			// n = tanh(InProj_n + r*hn)
			dan := dn * (1 - st.n[i]*st.n[i])
			dr := dan * st.hn[i]
			dhn := dan * st.r[i]
			g.InProj[base+2*H+i] += dan * invN
			g.BHH[2*H+i] += dhn * invN
			for j := 0; j < H; j++ {
				g.WHH[(2*H+i)*H+j] += dhn * st.hPrev[j] * invN
				dhNext[j] += dhn * p.WHH[(2*H+i)*H+j]
			}

			// r, z sigmoids
			dar := dr * st.r[i] * (1 - st.r[i])
			daz := dz * st.z[i] * (1 - st.z[i])
			g.InProj[base+i] += dar * invN
			g.InProj[base+H+i] += daz * invN
			g.BHH[i] += dar * invN
			g.BHH[H+i] += daz * invN
			for j := 0; j < H; j++ {
				g.WHH[i*H+j] += dar * st.hPrev[j] * invN
				g.WHH[(H+i)*H+j] += daz * st.hPrev[j] * invN
				dhNext[j] += dar*p.WHH[i*H+j] + daz*p.WHH[(H+i)*H+j]
			}
		}
	}
}

type adamState struct {
	m, v []float64
}

func newAdam(p *params) []*adamState {
	out := make([]*adamState, 0, 5)
	for _, n := range []int{len(p.InProj), len(p.WHH), len(p.BHH), len(p.WBit), len(p.BBit)} {
		out = append(out, &adamState{m: make([]float64, n), v: make([]float64, n)})
	}
	return out
}

func adamStep(w, g []float64, s *adamState, lr float64, t int, maxAbs float64) {
	const b1, b2, eps = 0.9, 0.999, 1e-8
	c1 := 1 - math.Pow(b1, float64(t))
	c2 := 1 - math.Pow(b2, float64(t))
	for i := range w {
		s.m[i] = b1*s.m[i] + (1-b1)*g[i]
		s.v[i] = b2*s.v[i] + (1-b2)*g[i]*g[i]
		w[i] -= lr * (s.m[i] / c1) / (math.Sqrt(s.v[i]/c2) + eps)
		if w[i] > maxAbs {
			w[i] = maxAbs
		} else if w[i] < -maxAbs {
			w[i] = -maxAbs
		}
	}
}

// Train fits a model to data and returns float weights plus progress.
func Train(data []byte, cfg Config) (*Result, *model.Weights, error) {
	cfg.defaults()
	if len(data) < cfg.SeqLen+8 {
		return nil, nil, fmt.Errorf("dataset too small (%d bytes) for seq_len=%d", len(data), cfg.SeqLen)
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	p := newParams(cfg.Hidden, rng)

	split := len(data) - int(float64(len(data))*cfg.Holdout)
	if split < cfg.SeqLen+8 {
		split = len(data) - cfg.SeqLen - 8
	}
	if split < 1 {
		split = len(data) / 2
	}
	tr := data[:split]
	windows := len(tr) - cfg.SeqLen - 2
	if windows < 1 {
		windows = 1
	}
	perEpoch := windows
	if cfg.StepsPerEpoch > 0 && perEpoch > cfg.StepsPerEpoch {
		perEpoch = cfg.StepsPerEpoch
	}
	if perEpoch > 20000 {
		perEpoch = 20000
	}

	adam := newAdam(p)
	states := make([]stepState, cfg.SeqLen)
	inputs := make([]byte, cfg.SeqLen)
	targets := make([]byte, cfg.SeqLen)
	res := &Result{}

	step := 0
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		var total float64
		ran := 0
		for i := 0; i < perEpoch; i++ {
			if cfg.MaxSteps > 0 && step >= cfg.MaxSteps {
				break
			}
			start := rng.Intn(windows)
			// Byte t is predicted from byte t-1; position 0 uses the start
			// token 0, exactly like the codec's first Step(0).
			inputs[0] = 0
			for k := 1; k < cfg.SeqLen; k++ {
				inputs[k] = tr[start+k-1]
			}
			copy(targets, tr[start:start+cfg.SeqLen])

			loss := p.forward(inputs, targets, states)
			g := p.newGrads()
			p.backward(inputs, targets, states, g)

			adamStep(p.InProj, g.InProj, adam[0], cfg.LR, step+1, 4)
			adamStep(p.WHH, g.WHH, adam[1], cfg.LR, step+1, 4)
			adamStep(p.BHH, g.BHH, adam[2], cfg.LR, step+1, 4)
			adamStep(p.WBit, g.WBit, adam[3], cfg.LR, step+1, 4)
			adamStep(p.BBit, g.BBit, adam[4], cfg.LR, step+1, 4)
			step++
			total += loss
			ran++
		}
		if ran == 0 {
			break
		}
		res.TrainBPB = append(res.TrainBPB, total/float64(ran))
		cfg.Logf("epoch %d/%d  train %.3f bits/byte", epoch+1, cfg.Epochs, total/float64(ran))
	}
	res.Steps = step

	if split+8 < len(data) {
		ev := data[split:]
		ei := make([]byte, len(ev)-1)
		et := make([]byte, len(ev)-1)
		ei[0] = 0
		for k := 1; k < len(ei); k++ {
			ei[k] = ev[k-1]
		}
		copy(et, ev[:len(et)])
		res.EvalBPB = p.forward(ei, et, nil)
		cfg.Logf("holdout: %.3f bits/byte (float model)", res.EvalBPB)
	}
	return res, p.Weights(1), nil
}

func (p *params) newGrads() *params {
	return &params{
		H:      p.H,
		InProj: make([]float64, len(p.InProj)),
		WHH:    make([]float64, len(p.WHH)),
		BHH:    make([]float64, len(p.BHH)),
		WBit:   make([]float64, len(p.WBit)),
		BBit:   make([]float64, len(p.BBit)),
	}
}
