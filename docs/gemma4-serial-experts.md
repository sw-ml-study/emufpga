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
request states are not combined. The later complete-logit and executable-code
experiments support that claim on bounded samples. Repository-scale coding
quality remains untested.

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
is broader coding tasks, finer residency tracing, and actual storage IO—not
connecting the model graph, which is now complete.

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

## Complete-path equivalence and residency audit

The logits probe ran one identical 32-token prompt twice through the same
patched llama.cpp binary and native Q5_K_M kernels. The control left selected
expert mmap pages resident; the candidate called `MADV_DONTNEED` after each
expert. All **262,144** final logits were bit-for-bit identical: zero differing
floats, zero maximum error, cosine similarity 1, and identical SHA-256 for both
1,048,576-byte arrays. This isolates the reclamation policy successfully for
one complete prompt path. It is not an independent implementation oracle.

A separately instrumented 1/2/4/8 coding-smoke sweep explains the surprising
RSS peak:

| Requests | Peak RSS MiB | File RSS MiB | Anonymous RSS MiB | Selected tensors | Logical expert GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,889 | 11,203 | 605 | 8,408 | 18.94 |
| 2 | 12,180 | 11,582 | 641 | 12,980 | 29.29 |
| 4 | 13,141 | 12,451 | 703 | 20,632 | 46.57 |
| 8 | 14,316 | 13,486 | 815 | 26,428 | 59.66 |

The peak is overwhelmingly file-backed mapped weights, not activations or an
anonymous copy of the model. Reclamation therefore has not produced the
one-expert theoretical resident set. More concurrent routes expand the union
of recently touched mappings. The next memory experiment should tighten mmap
access advice/readahead and measure cold-cache block-device traffic.

Adding `MADV_RANDOM` around each selected range preserved the bit-identical
logits. In a single follow-up run it changed peak RSS from 11,889 to 11,863 MiB
at one request (effectively unchanged) and from 14,316 to 13,514 MiB at eight.
The eight-request routes and operation count differed, so the 802 MiB reduction
is promising but **inconclusive**, not a claimed optimization. Repeated matched
runs are required. Raw-derived summary:
[`data/gemma4-q5km-madv-random.json`](data/gemma4-q5km-madv-random.json).

The small coding corpus produced expected answer text in 13/15 responses, but
obeyed the strict “return only the expression” prefix contract in 0/15. This
is a useful warning, not a coding-quality score: containment accepts prose and
the outputs were neither compiled nor tested. A sandboxed executable suite and
same-reference comparison remain required before claiming reliable coding
agents. Derived data:
[`data/gemma4-q5km-logit-equivalence.json`](data/gemma4-q5km-logit-equivalence.json)
and
[`data/gemma4-q5km-coding-residency.json`](data/gemma4-q5km-coding-residency.json).

## Executable Rust qualification

The follow-up replaced text containment with eight dependency-free Rust
function tasks. Tests were omitted from prompts, then each response was
compiled and executed in disposable Docker containers. Both compiler and test
runner had networking disabled, read-only roots, no Linux capabilities,
`no-new-privileges`, CPU/memory/PID limits, and timeouts. No generated code or
compile-time macro ran directly against the host.

Two repetitions at each concurrency produced 30 responses per policy:

| Requests | Resident passed | Reclaimed passed | Resident peak RSS | Reclaimed peak RSS | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2/2 | 2/2 | 12,118 MiB | 10,556 MiB | 4,150 MiB |
| 2 | 4/4 | 4/4 | 13,264 MiB | 10,836 MiB | 4,150 MiB |
| 4 | 8/8 | 8/8 | 14,376 MiB | 11,163 MiB | 4,152 MiB |
| 8 | 16/16 | 16/16 | 15,165 MiB | 12,962 MiB | 4,152 MiB |

| Requests | Resident mean inference | Reclaimed mean inference | Resident logical GB | Reclaimed logical GB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.75 s | 16.52 s | 138.49 | 138.49 |
| 2 | 10.59 s | 22.47 s | 223.14 | 223.14 |
| 4 | 16.92 s | 34.90 s | 425.73 | 398.55 |
| 8 | 27.57 s | 52.20 s | 641.40 | 634.13 |

All 60 generated programs compiled and passed. Across 30 paired cases there
were zero pass/fail disagreements and 28 exact-text matches. The two text
differences were both palindrome implementations at concurrency eight; both
passed. Strict function-only formatting was only 8/30 for each policy, showing
that textual compliance is unstable even when the executable result is sound.

This is positive bounded reliability evidence, not proof that several coding
agents can complete repository work at equal quality. The corpus is small, the
tests are deterministic, and there are only two repetitions. Reclamation also
increased mean inference wall time; speed is a usability cost, not the capacity
success criterion. NVIDIA-board telemetry integrated 4.35 kJ resident and
6.60 kJ reclaimed, but the sampling window included sequential compiler/test
containers after inference. Those values are disclosed but are not a valid
inference-energy comparison or results/kWh claim. Derived data:
[`data/gemma4-q5km-executable-code.json`](data/gemma4-q5km-executable-code.json).
The physical-I/O follow-up is predeclared in
[`cold-cache-io-plan.md`](cold-cache-io-plan.md).

That qualification is now complete at concurrency 1/2/4/8 with three cold and
three warm repetitions per policy. All 180 executable Rust evaluations passed.
Cold request reads grew from 8.12 GB at c1 to only 10.70 GB at c8, so bytes per
passing task fell from 8.12 to about 1.34 GB. Resident completion rate rose from
23.2 to 109.8 tasks/hour; reclaimed rose from 21.2 to 93.2. Reclamation saved
1.47--2.79 GiB mean peak RSS depending on concurrency but did not reduce disk
bytes. All physical request reads averaged approximately 4 KiB. Consequently,
this validates shared demand-paged reuse and the memory/latency tradeoff, not a
purpose-built sequential expert stream. See the checked-in
[c1/c8](data/gemma4-q5km-cold-hdd-c1-c8-r3.json) and
[c2/c4](data/gemma4-q5km-cold-hdd-c2-c4-r3.json) records.

### Does this fit an 18 GB unified-memory Mac?

Not safely with the measured residency policy. On this discrete-memory Linux
host, the 8-request peak combines about 14.0 GiB process RSS with about 4.1 GiB
VRAM. Those pools overlap differently on Apple unified memory, so simply adding
them is not a prediction; nevertheless, their roughly 18.1 GiB sum leaves no
space for macOS, Metal allocations, page tables, or other applications. Even
the one-request 11.6 GiB RSS plus GPU-resident state would be tight.

The model file itself may remain on SSD and mmap does permit reclaimable pages,
so 18 GB is not ruled out architecturally. It requires a materially smaller
file-backed working set, a one-request experiment first, and macOS measurements
of resident/compressed memory and swap. The current data supports “plausible
after residency repair,” not “expected to fit.”
