# Architecture

`nzip` is a two-plane compressor: a Python/PyTorch trainer that fits a small
byte-level GRU to one schema, and a zero-dependency Go codec that runs it across
CPU cores. Everything below is implemented; the numbers are measured on the
corpora named in each section.

```
                    control plane (Python or pure Go)
  raw samples ──▶  train ──▶ evaluate with the *Go codec* ──▶ publish ──▶ promote
  (spool)                     on a holdout it never saw        (content-   (per-dataset
                                                               addressed)  manifest)
                                        │ NGRM artifact (id + crc32c)
                                        ▼
                    data plane (Go, no dependencies)
  streams ──────▶  NGR3/NGR6 frames: crc32c, model hash, profile, keyframes
                    integer GRU · match model · order-2/3/4 · word · run-length
                    8 binary decisions per byte, binary arithmetic coder
```

## Packages

| package | responsibility |
| :--- | :--- |
| `cmd/nzip` | CLI: `compress`, `decompress`, `bench`, `eval`, `ablate`, `verify`, `inspect`, `train`, `pipeline`, `models` |
| `codec` | the hot path: integer GRU, match model, context models, mixer, APM, arithmetic coder, spatial transforms |
| `model` | `NGRM` artifact parsing, quantisation, content hash, registry with hot-swap |
| `stream` | framing, per-block profiles, `blocks`/`session` modes, dictionary, keyframes, `io.Reader`/`io.Writer` adapters |
| `drift` | EWMA bits-per-byte monitor and retrain trigger |
| `store/fs` | content-addressed artifacts plus a per-dataset promotion manifest |
| `internal/train` | dependency-free trainer, so the whole loop runs in CI |

## Data plane

### Per-byte pipeline

```
prev byte ──▶ quantised GRU ─┐
4/8/16-byte ─▶ match model ──┤
order-2 ─────────────────────┤──▶ logistic mixer ──▶ SSE/APM ──▶ binary
order-3/4, word (text only)  │      (per node, run,        (per node,    arithmetic
run-length, class ───────────┘       match state)           match state)  coder
```

Each byte is **eight binary decisions** through a bit tree. That replaced a
256-way softmax (`256·H` multiply-adds plus 256 `exp` per byte) with `8·H`, and
removed the 1/65536 confidence floor that the old 256-bin CDF imposed.

### Why it is integer-only

Go may contract `a*b + c` into a single fused multiply-add; on arm64 it does, on
amd64 it does not. With float weights, encoding on one architecture and decoding
on the other corrupted **3869 of 4096 bytes**. Everything on the hot path is
therefore fixed point — Q16 probabilities, Q7 log-odds, Q11 activations and
weights, int32 accumulators — with `exp`/`sigmoid`/`tanh` as LUTs built from
integer arithmetic. The SSE2 kernel (`PMADDWD`) and the pure-Go fallback perform
the same integer arithmetic, so they are interchangeable by construction.

**Accumulator bound:** `H ≤ 96` and weights clamped to ±4 give
`96 × 8192 × 2048 = 1.6e9 < 2^31`. Hidden dimensions above that are rejected at
load.

### Determinism and what may float

Encoder and decoder must derive identical probabilities. Consequently:

* all model state is integer;
* table sizes are a **function of the profile**, never of the machine (a decoder
  with a different worker count must still agree);
* the only floating point is the encoder-side choice of image transform, which
  is recorded in the frame header and therefore never recomputed by the decoder.

## Automatic profiles

One configuration cannot serve every kind of data, so each block is classified
from cheap byte statistics and the decision travels in the frame header.

| profile | chosen when | configuration |
| :--- | :--- | :--- |
| `telemetry` | default; printable with markup or binary without a raster | order-2 (2^18), run-length, match |
| `text` | mostly printable, ≥ 8 % spaces | adds order-3/4 + word model (2^20 each) |
| `image` | non-ASCII > 25 % **and** a spatial predictor that lowers residual entropy | row delta or Paeth, stride from the data |

Word spacing, not markup density, separates prose from minified JSON: Wikipedia
is full of XML tags and still wants the deeper context. The image transform is
chosen by measured residual entropy, so noise and already-compressed data are
left alone; the stride is discovered from the data because a block in the middle
of a file has no header.

Measured effect of classification: text −15 %, images −24 %, telemetry −3 %
(shared SSE stage).

## Stream format

See [format.md](format.md) for the byte layout. The invariants that matter:

* **integrity** — crc32c over the header and every frame, so corruption fails
  loudly instead of decoding to plausible garbage;
* **model identity** — the stream records the artifact's content hash, so
  retraining model 101 never breaks yesterday's archives;
* **bounded allocation** — every length is validated against a hard cap before
  memory is reserved;
* **resumability** — `session` mode emits keyframes (hidden state + 64 KiB
  history, ~65 KiB) when they cost less than a tenth of the bytes between them;
* **no silent expansion** — incompressible blocks are passed through unchanged.

## Control plane

`nzip pipeline` implements the loop end to end:

1. read sampled raw batches from a spool directory;
2. train (pure Go, or substitute PyTorch via `train.py`, same architecture);
3. **evaluate with the real codec** on a holdout the trainer never saw — the
   gate is the quantised codec's bits-per-byte, not the training loss;
4. publish the artifact content-addressed;
5. promote it for the dataset only if it beats the current model by a margin.

Old artifacts are never deleted, which is what makes step 5 safe.

`drift.Monitor` watches the same stream as an observer: the baseline is an EWMA,
upward adaptation is 4× slower than downward so a genuine regression is still
reported, and `nzip compress -drift 0.2` surfaces alerts.

## Measured results

| corpus | nzip | bzip2 -9 | xz -9e | gzip -9 |
| :--- | ---: | ---: | ---: | ---: |
| OTel JSON, 7.4 MB | **0.0823** | 0.0865 | 0.0902 | 0.1065 |
| Wikipedia, 20 MB | **0.2663** | 0.2883 | 0.2621 | 0.3664 |
| enwik9, 1 GB (before the last table change) | 0.2495 | 0.2540 | 0.2118 | 0.3226 |
| 20 generated images (PNG + BMP) | **0.6034** | 0.7402 | 0.7309 | 0.8513 |

Throughput: ~11–12 MB/s on telemetry at 32 threads (≈1.2 MB/s/core), ~2.5 MB/s
on text with the deep tables. `nzip` beats bzip2 on every corpus measured.

For scale, the Hutter Prize record on enwik9 is 0.1004 (a cmix-class ensemble
at roughly 1 KB/s and ~10 GB RSS). That is a different point on the curve, not a
competitor.

## Design rules learned the hard way

**Predictor count saturates before predictor quality does.** Every accepted
change either added a *kind* of information or removed a *bottleneck*; every
rejected change added another view of information the mixer already had.

| change | effect |
| :--- | ---: |
| bit-tree alphabet + integer LUTs (vs 256-way softmax) | **29× faster** |
| multi-length match candidates | match-only −16 %, final −0.06 % |
| mixer address = (node, run length) | −3.3 % |
| order-2 context (vs order-0.5) | −4.4 % |
| context table 2^14 → 2^16 | −2.6 % |
| mixer address keyed by match state | −3.2 % |
| SSE/APM stage | −2.8 % telemetry, −2.6 % text |
| auto image transform (delta or Paeth) | images 0.648 → 0.603 |
| order-4 + word model (text) | −2.5 % |
| table 2^16 → 2^18, later 2^20 for text | −7.9 %, then −5.2 % |
| second match candidate | **+0.2 %** |
| record-aligned predictor | **+0.4 %** |
| count/confidence channel for deep contexts | **+0.04 %** |
| deep contexts for telemetry | **+0.4 %** |
| order-5 + skip context | **+0.9 %** |

Two bugs in that list are worth remembering: deeper contexts must be indexed by
the bit-tree node **inside** the bit loop (indexing once per byte turned a 3 %
win into a 7 % loss), and every new predictor must not start at weight 0.5 or the
mixed logit becomes over-confident.

## Verification

`gofmt`, `go vet`, unit and round-trip suites, `-race`, `-tags purego` (scalar
path), `GOARCH=arm64` compile, and a fuzz target over the decompressor. The
suites cover: coder edge cases, corruption and truncation detection, hostile
lengths, determinism across worker counts, session resume from a keyframe,
mixed-profile streams in both modes, transform reversibility, classifier
decisions, and APM convergence.
