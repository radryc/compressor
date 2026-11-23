package codec

// Binary arithmetic coder (carry-less, 32-bit range, 16-bit probabilities).
//
// One binary decision is coded per call. A byte therefore costs eight
// EncodeBit calls instead of building and searching a 256-entry CDF, which is
// both faster and more precise: probabilities are never quantised down to a
// 1/65536 bucket per symbol.

const (
	maxRange      = uint32(0xFFFFFFFF)
	half          = uint32(0x80000000)
	quarter       = uint32(0x40000000)
	threeQuarters = uint32(0xC0000000)
)

type ArithmeticEncoder struct {
	buf      []byte
	low      uint32
	high     uint32
	pending  int
	curByte  byte
	bitCount uint8
}

func (e *ArithmeticEncoder) Reset(dst []byte) {
	e.buf = dst[:0]
	e.low = 0
	e.high = maxRange
	e.pending = 0
	e.curByte = 0
	e.bitCount = 0
}

func (e *ArithmeticEncoder) writeBit(b byte) {
	e.curByte = (e.curByte << 1) | (b & 1)
	e.bitCount++
	if e.bitCount == 8 {
		e.buf = append(e.buf, e.curByte)
		e.curByte = 0
		e.bitCount = 0
	}
}

func (e *ArithmeticEncoder) emitBit(b byte) {
	e.writeBit(b)
	inv := 1 - b
	for i := 0; i < e.pending; i++ {
		e.writeBit(inv)
	}
	e.pending = 0
}

// EncodeBit codes bit with probability p = P(bit=1) in Q16.
//
// The "1" subinterval is the lower part of the range:
//
//	bit=1 -> [low, low+p1-1]      p1 = rng*p/65536
//	bit=0 -> [low+p1, high]
func (e *ArithmeticEncoder) EncodeBit(p uint32, bit uint32) {
	rng := uint64(e.high) - uint64(e.low) + 1
	p1 := uint32((rng * uint64(clampProb(p))) >> ProbBits)
	if bit != 0 {
		e.high = e.low + p1 - 1
	} else {
		e.low += p1
	}
	for {
		if e.high < half {
			e.emitBit(0)
		} else if e.low >= half {
			e.emitBit(1)
			e.low -= half
			e.high -= half
		} else if e.low >= quarter && e.high < threeQuarters {
			e.pending++
			e.low -= quarter
			e.high -= quarter
		} else {
			break
		}
		e.low <<= 1
		e.high = (e.high << 1) | 1
	}
}

func (e *ArithmeticEncoder) Finish() []byte {
	e.pending++
	if e.low < quarter {
		e.emitBit(0)
	} else {
		e.emitBit(1)
	}
	if e.bitCount > 0 {
		e.curByte <<= (8 - e.bitCount)
		e.buf = append(e.buf, e.curByte)
	}
	return e.buf
}

type ArithmeticDecoder struct {
	data    []byte
	bytePos int
	low     uint32
	high    uint32
	value   uint32
	bitPos  uint8
}

func (d *ArithmeticDecoder) Reset(data []byte) {
	d.data = data
	d.bytePos = 0
	d.bitPos = 0
	d.low = 0
	d.high = maxRange
	d.value = 0
	for i := 0; i < 32; i++ {
		d.value = (d.value << 1) | d.readBit()
	}
}

func (d *ArithmeticDecoder) readBit() uint32 {
	if d.bytePos >= len(d.data) {
		return 0
	}
	b := (d.data[d.bytePos] >> (7 - d.bitPos)) & 1
	d.bitPos++
	if d.bitPos == 8 {
		d.bitPos = 0
		d.bytePos++
	}
	return uint32(b)
}

// DecodeBit returns the next bit given p = P(bit=1) in Q16.
func (d *ArithmeticDecoder) DecodeBit(p uint32) uint32 {
	rng := uint64(d.high) - uint64(d.low) + 1
	p1 := uint32((rng * uint64(clampProb(p))) >> ProbBits)
	var bit uint32
	if d.value-d.low < p1 {
		bit = 1
		d.high = d.low + p1 - 1
	} else {
		bit = 0
		d.low += p1
	}
	for {
		if d.high < half {
		} else if d.low >= half {
			d.low -= half
			d.high -= half
			d.value -= half
		} else if d.low >= quarter && d.high < threeQuarters {
			d.low -= quarter
			d.high -= quarter
			d.value -= quarter
		} else {
			break
		}
		d.low <<= 1
		d.high = (d.high << 1) | 1
		d.value = (d.value << 1) | d.readBit()
	}
	return bit
}
