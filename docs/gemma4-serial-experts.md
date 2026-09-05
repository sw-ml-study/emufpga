# Gemma-4 Q5_K_M selected-expert stream

## Concise conclusion

The real oversized Gemma artifact can now route layer-0 activations, copy only
the selected experts into expert-ID order, and execute those fixed Q5_K/Q8_0
bytes with 176 bytes of parameter payload resident in the stream reader. At
batch 8, 64 expert assignments collapse to 33 expert fetches and the streamed
result differs from the direct GGUF path by at most 0.00000381.

This validates the layer mechanism and same-quant arithmetic. A later section
records complete text generation through the experimental mmap-backed path. It
does **not** yet demonstrate coding quality. The standalone scalar stream loop
is 2.6–3.5 times slower than the direct batched Rust layer oracle.

## Measured layer-0 result

The input is deterministic synthetic activation data. Gemma's real float32
router selects top 8 of 128 experts. Both paths use exact expert bytes from the
pinned 19.3 GB Q5_K_M GGUF: Q5_K fused gate/up and Q8_0 down projection.

| Concurrent activations | Assignments | Distinct fetches | Uses/fetch | Stream MB | MB/request | Direct ms | Serial ms | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 1.00 | 41.14 | 41.14 | 144 | 373 | 0.00000191 |
| 2 | 16 | 12 | 1.33 | 61.70 | 30.85 | 225 | 608 | 0.00000381 |
| 4 | 32 | 20 | 1.60 | 102.84 | 25.71 | 387 | 1,099 | 0.00000381 |
| 8 | 64 | 33 | 1.94 | 169.69 | 21.21 | 690 | 2,447 | 0.00000381 |

Amortization is `assignments / distinct experts`. Eight requests nearly double
applications per fetch and cut stream bytes per request by 48%. They do not run
at the latency of one: routes expand the union from 8 to 33 experts, and each
assignment still needs its own matrix arithmetic.

The `.spm` stream uses about 5.14 MB per selected expert versus 4.83 MB raw
GGUF payload because every quant block carries framing. That 6.4% overhead is
removable. It is not residency: the reader holds one 176-byte Q5_K block at a
time, plus activations and outputs.

## What this says about several coding agents

If three agents reach a layer together, the sample supports useful expert
reuse—but not three agents at the latency of one. A scheduler fetches each
union expert once and applies it to every waiting request that selected it.
Attention, KV cache, sampling, and matrix arithmetic remain per request.

Quality should be unchanged in principle because bytes are not requantized and
request states are not combined. This layer comparison supports that claim
numerically. Reliability and coding quality still require complete greedy
generation to match a same-quant llama.cpp oracle and the coding task suite.

No defensible maximum agent count exists yet. Route-union growth, KV capacity,
latency, and the batching window jointly set it. The end-to-end experiment must
test 1/2/4/8 independent requests. Those rates characterize usability and
amortization; they are not conditions for the primary capacity claim.

## Reproduce

```console
scripts/verify-gemma4-expert-smoke 1
scripts/verify-gemma4-expert-smoke 2
scripts/verify-gemma4-expert-smoke 4
scripts/verify-gemma4-expert-smoke 8
```

The wrapper pins SHA-256 and writes generated streams under `/disk1/tmp`.
Derived data: [`data/gemma4-q5km-serial-layer0.json`](data/gemma4-q5km-serial-layer0.json).

## Integration boundary

llama.cpp tensor placement alone is not serial selected-expert execution. The
experiment below therefore adds a narrow native-operation patch: lazy expert
mappings plus post-expert page reclamation. The remaining validation boundary
is reference logits, coding tasks, finer residency tracing, and actual storage
IO—not connecting the model graph, which is now complete.

## End-to-end partition bridge

A one-run llama.cpp smoke placed every `ffn_*_exps.weight` tensor on CPU while
requesting all repeating layers on GPU. It completed all 1/2/4/8 short tasks
correctly. Peak VRAM was 4,170 MiB; peak process RSS was 19,514.5 MiB.
Aggregate generation was 2.51, 2.92, 4.97, and 5.66 tok/s respectively.

This proves that GPU attention/KV/shared work plus CPU expert work is a viable
complete-model partition. It also rejects an easy shortcut: mapped tensor
override consumes approximately full-model host residency, so it is not the
bounded stream and likely does not fit an 18 GB unified-memory Mac alongside
the OS and runtime. Derived data:
[`data/gemma4-q5km-experts-cpu-smoke.json`](data/gemma4-q5km-experts-cpu-smoke.json).

## First bounded end-to-end result

The checked-in llama.cpp patch marks Gemma expert tensors for lazy mmap access,
retains the native `MUL_MAT_ID` quantized kernels, processes selected experts
in expert-ID order, and applies Linux `MADV_DONTNEED` after all worker threads
finish an expert. It does not requantize weights or replace the model graph.

Across three runs of the 128-input/16-output smoke contract:

| Concurrent requests | Correct | Aggregate tok/s | Per-request tok/s | TTFT p50 | TTFT p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3/3 | 1.80 | 1.80 | 6.76 s | 8.82 s |
| 2 | 6/6 | 2.50 | 1.25 | 9.81 s | 12.85 s |
| 4 | 12/12 | 3.66 | 0.92 | 11.19 s | 17.37 s |
| 8 | 24/24 | 4.80 | 0.60 | 20.06 s | 21.04 s |

All-GPU allocation still fails. The bounded prototype used 4,172 MiB peak
VRAM. Process RSS ranged from 2,612 to 15,070 MiB and averaged 9,723 MiB,
versus 19,514.5 MiB peak for conventional CPU expert placement. Residency is
therefore reclaimable and 4,444.6 MiB lower at peak, but it is not yet close to
the one-expert theoretical working set. Large prompt batches touch wide expert
unions before pages can be reclaimed.

Native tracing recorded 11,700 expert matrix operations and 465.09 GB of
logical selected-expert bytes over server load plus the complete three-run
sweep. Logical bytes are not physical SSD reads: clean pages may remain in the
kernel page cache after process mappings are advised away.

NVIDIA-board energy was 4,912 J over 200.13 seconds. CPU, RAM, SSD, fans, PSU
loss, and idle baseline are excluded, so this is not a whole-system efficiency
claim.

This is a **preliminary capacity success**: a model that fails resident GPU
allocation generated correct short answers through reclaimable selected expert
pages. It is not a coding-reliability result. Generated continuations can vary
between CPU/GPU placements despite temperature zero, so the next correctness
gate needs reference logits plus a real coding task corpus rather than exact
text identity.

Derived data:
[`data/gemma4-q5km-bounded-end-to-end.json`](data/gemma4-q5km-bounded-end-to-end.json).
The reproducible patch is
[`patches/llama.cpp-gemma4-lazy-experts.patch`](../patches/llama.cpp-gemma4-lazy-experts.patch).
