# Building a training corpus

The model is byte-level: it learns the exact bytes it will later compress. A
corpus that differs from production serialisation — whitespace, key order,
JSON vs protobuf — costs ratio directly. There is no normalisation step at
compression time, by design.

## 1. Pick the wire format, then freeze it

Compress the bytes the ingest tier already holds.

| Path | Bytes on the wire | Recommendation |
| :--- | :--- | :--- |
| OTLP/gRPC (`:4317`) | protobuf | train on protobuf if you compress here |
| OTLP/HTTP (`:4318`) protobuf | protobuf | train on protobuf |
| OTLP/HTTP JSON | JSON | train on the *exact* JSON your exporter emits |
| converted / normalised JSON | re-encoded | do not mix with either of the above |

A model trained on minified JSON will be close to useless on protobuf and vice
versa. Pick one and record the choice alongside the artifact.

If you use JSON:

* minified, one `ExportTraceServiceRequest` per line (NDJSON), `\n` terminated;
* no BOM, strict UTF-8;
* deterministic key order (protojson field order, or sorted struct fields);
* no wrapping array, so historical dumps concatenate with `cat`.

## 2. What the model can and cannot learn

| Field | Behaviour |
| :--- | :--- |
| structural keys, repeated envelopes | first the match model, then the GRU, drive these to near-zero bits |
| repeated service names, scopes, attribute keys | high predictability; both models compress heavily |
| `traceId` / `spanId` | 32 / 16 hex chars of real entropy — ~4 bits/char is the floor |
| `startTimeUnixNano` | shared high-order digits compress; low-order digits do not |
| `kind`, `status.code` | tiny alphabet, fractions of a bit |

Consequences:

1. keep real (or cryptographically random) identifiers — never dummy `0000…`,
   or the model learns 99.9 % confidence in `'0'` and pays for it in production;
2. keep the batching granularity the same (spans per envelope, bytes per batch);
3. separate corpora per signal (traces, logs, metrics) and per tenant if their
   attribute sets differ.

## 3. Size and split

* 10–50 MB of representative NDJSON, or 5–20 MB of protobuf, is enough for
  `H=32`.
* Hold out the **last 5–10 %** and never train on it.
* The only number that matters is the one the Go codec produces on that
  holdout:

```bash
nzip train -data corpus.bin -out models/101.bin -model-id 101 \
           -hidden 32 -epochs 6 -seq-len 64 -holdout 0.05

nzip eval  -model models/101.bin -in holdout.bin
nzip ablate -model models/101.bin -in holdout.bin      # is the GRU earning its CPU?
nzip bench -model models/101.bin -in holdout.bin       # against gzip -9
```

## 4. Choosing the hidden size

| H | per-core speed | use |
| --- | --- | --- |
| 16 | ~2.5 MB/s | maximum throughput, homogeneous schemas |
| 32 | ~1.4 MB/s | default |
| 64 | ~0.67 MB/s | archival only, many distinct attribute sets |

`H > 96` is rejected: the quantised accumulator is only proven to fit in int32
up to that size (see `docs/format.md`).

## 5. Retraining

Promote through `nzip pipeline`, not by overwriting a file:

```bash
nzip pipeline -spool /var/lib/nzip/spool \
              -models /var/lib/nzip/models \
              -dataset otel-traces \
              -hidden 32 -epochs 6
```

It trains, evaluates with the real codec on a holdout, publishes the artifact
content-addressed, and promotes it for the dataset only if it beats the current
model by `-min-improvement` (default 2 %). Old artifacts are never deleted, so
archives written before the retrain still decode.

Sample 1 % of raw batches into the spool (or up to a byte budget per day) and
keep them short-lived; the spool is the only place raw, uncompressed telemetry
should live.
