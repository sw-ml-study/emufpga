# Packed Q6_K MoE streaming benchmark

This experiment keeps Granite 3.1 1B-A400M expert weights in their original
GGML Q6_K representation inside `.spm`. A Q6_K group contains 256 weights in
210 bytes. The Rust engine reads one group, decodes it into 256 `f32` values,
uses it for every routed token in the batch, and then reuses that storage for
the next group.

The source model is
`granite-3.1-1b-a400m-instruct-Q6_K.gguf`. Results below are block 0, release
build, warm Linux filesystem cache, on the local test host. They are
representative single runs, not claims about storage-device or GPU throughput.
Run `scripts/bench-granite-q6` to reproduce them.

## Capacity and correctness

| Measure | Packed Q6_K | Previous f32 `.spm` |
|---|---:|---:|
| all-expert file | 42,212,416 bytes | 201,792,576 bytes |
| compression versus f32 | 4.78x smaller | baseline |
| streams | 97 | 97 |
| packed group residency | 210 bytes | 4,096 bytes |
| peak input path | 132,306 bytes | about 136 KiB |
| rewinds | 0 | 0 |
| maximum expert error | 0.00000477 | 0.00000334 |
| maximum combined error | 0.00000191 | 0.00000095 |

Peak input is two 64 KiB file buffers, the 210-byte packed group, and the
1,024-byte decoded group. It excludes activations and output accumulators,
which are working state rather than parameter residency.

The expert blocks are copied without requantization. They remain row-major,
so the Q6_K executor maps each decoded position to `(row, column)` rather than
using the column-major dense executor. Results agree with both the direct GGUF
Rust path and the established llama.cpp-derived oracle.

## Measured batch sweep

All times exclude the router and include the expert sweep only. `tokens/s` is
`batch / (read + decode + compute)`. Read GB/s uses physical `.spm` bytes and
is approximate because the small router/prologue is outside the expert timer.

| Schedule | Batch | Unique experts | File bytes | Emit ms | Read ms | Decode ms | Compute ms | tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1 | 8 | 42,212,416 | 29.8 | 13.4 | 8.6 | 54.8 | 13.0 |
| all | 4 | 19 | 42,212,416 | 30.7 | 14.5 | 21.7 | 183.2 | 18.2 |
| all | 8 | 28 | 42,212,416 | 32.6 | 15.8 | 32.7 | 363.1 | 19.4 |
| all | 16 | 28 | 42,212,416 | 29.8 | 15.7 | 33.0 | 696.9 | 21.5 |
| all | 32 | 28 | 42,212,416 | 131.6 | 16.6 | 35.5 | 1,174.5 | 26.1 |
| selected union | 1 | 8 | 10,654,528 | 10.4 | 4.2 | 8.8 | 64.7 | 12.9 |
| selected union | 4 | 19 | 25,118,560 | 18.0 | 9.7 | 21.7 | 190.2 | 18.0 |
| selected union | 8 | 28 | 36,952,768 | 131.0 | 14.7 | 34.5 | 382.4 | 18.5 |
| selected union | 16 | 28 | 36,952,768 | 135.7 | 14.4 | 34.7 | 672.6 | 22.2 |
| selected union | 32 | 28 | 36,952,768 | 133.7 | 14.8 | 35.6 | 1,234.1 | 24.9 |

Emission variance is visibly large once the operating system starts writing
back dirty pages. It is intentionally reported rather than folded into
inference. Repeated trials and percentiles belong in the next benchmarking
step.

## What the numbers mean

For one token, an all-expert sweep moves 42.2 MB while only 10.65 MB is useful:
25.2 percent stream utilization. At batch 4, 19 experts are used and utilization
rises to 59.5 percent. This token sample reaches 28 of 32 experts by batch 8,
so eliminating unused experts can save only about 12.5 percent after that.

The selected-union layout is a route-specialized comparison. It proves how
many bytes could be saved when routes are known before laying out or fetching
the expert region. The all-expert layout remains the clean route-independent,
strictly forward-only baseline.

Read time falls when unused experts are removed, especially at batch 1. Total
CPU time barely improves because scalar matrix arithmetic dominates this
prototype. On a faster GPU or FPGA, transfer becomes a larger fraction and
the saved bytes matter more. In simple terms:

```text
elapsed time >= max(bytes / bandwidth, arithmetic / compute rate)
```

Q6_K attacks the first term. Batching attacks both terms by reusing each
decoded weight across multiple tokens. Vectorized CPU, CUDA, or FPGA kernels
are needed to reduce the second term.

## Cold-read limitation

Cold reads are reported as `unmeasured`. Safely forcing a true cold read would
require evicting filesystem cache globally or reading a data set larger than
cache, either of which perturbs unrelated work on this shared machine. A later
benchmark should use direct I/O or a dedicated disposable device if cold-media
bandwidth is important.
