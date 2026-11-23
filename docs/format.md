# Formats

Two artifacts: `NGRM` (model weights) and `NGR3` (compressed stream). Both are
little-endian for multi-byte fields.

## NGRM v2 — model artifact

```
offset  size  field
0       4     magic "NGRM"
4       2     version (2)
6       2     hidden dimension H (1..96)
8       4     model id
12      4     reserved
16      4*N   float32 weights
```

Weights are stored as float32 in exactly this order:

| slice | shape | use |
| :--- | :--- | :--- |
| `InProj` | `256 × 3H` | input projection for each possible previous byte, input bias folded in |
| `WHH` | `3H × H` | recurrent weights, gates ordered **r, z, n** |
| `BHH` | `3H` | recurrent bias |
| `WBit` | `8 × H` | one row per bit of the next byte (MSB first) |
| `BBit` | `8` | bit-head bias |

`N = 256*3H + 3H*H + 3H + 8H + 8`.

The codec quantises on load:

* `InProj`, `BHH` → int32 Q16 (saturating),
* `WHH`, `WBit` → int16 Q11, clamped to ±4.0,
* `BBit` → int32 Q7.

Quantisation uses `math.Round(float64(f) × 2^k)`, which is exact for powers of
two, so the same file always yields the same integers.

**Content hash** is crc32c over every byte of the artifact. The stream records
it, so a retrained model id cannot silently break an old archive.

**Invariant:** `H ≤ 96` together with the ±4 weight clamp bounds every dot
product by `96 × 8192 × 2048 = 1.61e9 < 2^31`. Above that the int32 accumulator
would wrap (deterministically, but with a large ratio loss), so `H > 96` is
rejected at load.

## NGR4 — stream

### Header (32 bytes)

```
0   4  magic "NGR3"
4   1  version (5)
5   1  flags: bit0 session, bits1-4 blend (GRU, match, order-2, run-length)
6   2  header length (32)
8   4  model id
12  4  model content hash (crc32c of the artifact)
16  4  block size (1..16 MiB)
20  4  frame payload cap (1..64 MiB)
24  4  dictionary hash (0 = none)
28  4  crc32c of bytes [0:28]
```

### Frame header (24 bytes) + payload

```
0   8  sequence number
8   4  raw length
12  4  compressed length
16  1  flags: 0x01 passthrough, 0x02 keyframe, 0x04 dictionary, 0x80 end
17  1  profile: 0 telemetry, 1 text, 2 image
18  2  row stride for the image profile (0 = no transform)
20  4  crc32c of bytes [0:20] followed by the payload
```

Frame kinds:

| kind | raw length | meaning |
| :--- | :--- | :--- |
| coded | `n > 0` | `n` bytes were arithmetic-coded |
| passthrough | `n = compLen` | the block did not compress; payload is the original bytes. The decoder must still advance the model over them (`Prime`), so session state stays in sync |
| dictionary | `0` | priming bytes; not part of the output; must be the first frame |
| keyframe | `0` | hidden vector, previous byte and a 64 KiB history window; adaptive tables are reset on both sides |
| end | `0` | terminator |

### Validation before allocation

The decoder rejects, without allocating:

* `blockSize` outside `1..16 MiB`, `frameMax` outside `1..64 MiB`;
* `rawLen > blockSize` (except passthrough, which is checked separately);
* `compLen > frameMax`;
* `compLen > 16 × rawLen + 8` — the arithmetic coder cannot expand more than
  16 bits per bit with a Q16 probability;
* keyframes larger than 1 MiB;
* any checksum mismatch.

### Versioning

The version is bumped to **5** because the frame header now carries a per-block
profile and stride, and because version 4's probability model changed (order-2
and run-length contexts, match-state mixer addressing). An archive compressed by an
older codec would decode to different bytes, and the frame CRC only covers the
payload, so the version check is the only thing that stops a silent
mistranslation.

### Per-block profiles

Each data frame records which codec configuration and transform the encoder
used, so the decoder never guesses:

| profile | transform | codec |
| :--- | :--- | :--- |
| 0 telemetry | none | order-2 + run + match |
| 1 text | none | order-2 + **order-3** + run + match |
| 2 image | `p[i] -= p[i-stride]` | order-2 + run + match |

Passthrough frames always carry the original bytes and therefore always have
`stride = 0`. A stride on a non-image profile, or on a passthrough frame, is
rejected as malformed.

### Session mode and keyframes

Session mode keeps the GRU hidden state, adaptive statistics and match history
across frames. A keyframe is emitted when both conditions hold: at least
`keyframe interval` raw bytes (default 8 MiB) have been coded since the last
one, **and** the keyframe would cost at most a tenth of the bytes stored since
the last keyframe. On very compressible streams that means few or no keyframes —
such streams are cheap to replay anyway. A keyframe contains:

```
u16 snapshot version (2), u16 H, u8 blend, u32 prev byte, u32 history length,
H × int16 hidden state, history bytes
```

The adaptive tables (2^16 bit statistics, 2^12 run statistics, 4096×5 mixer
weights) are **not** serialised. A snapshot is ~65 KiB instead of ~520 KiB, and
the writer rewinds its own state to the snapshot after emitting it, so both sides
continue from identical state.

The compressor trims its match history to the same 64 KiB window before
snapshotting, so encoder and decoder agree: the decoder rebuilds the match model
by replaying the window, which makes resumption deterministic. `nzip decompress -start-seq N` skips to the
first keyframe at or after `N`; `stream.Reader` supports the same via
`WithStartSeq`.

### Dictionary

A dictionary frame is raw bytes that both sides feed through the model before
coding anything. It is not part of the output. The header carries its crc32c so
a mismatched dictionary is rejected rather than silently producing garbage.
