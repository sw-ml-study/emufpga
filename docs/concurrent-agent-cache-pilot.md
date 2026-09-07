# Concurrent-agent native cache pilot

## Concise conclusion

The operation-shared implementation is correct and removes the accidental
per-worker cache traffic: one selected-expert operation now causes one acquire
and one release, regardless of CPU worker count. A fresh deterministic
three-step run produced byte-identical logits with and without the cache.

The rerun does **not** yet show a production win. The 512 MiB cache copied 10.1%
fewer logical source bytes than the 1 MiB no-admission server, but dynamic
batching differed between servers and therefore accounts for some unknown part
of that difference. The candidate was 17% to 61% slower across the measured
concurrency levels. Copy-on-admission and eviction churn, not a per-worker
mutex, are now the dominant measured costs.

These are single CPU-only pilots, not a validated project claim.

## Operation-shared rerun (2026-09-07)

Both servers used the same c4, c1, c2, c8 request order and generated 16 tokens
per request. All response text matched between policies. The short limit
truncated every function before it compiled, so this run provides a
determinism check, not a coding-quality score.

| Metric | 1 MiB no admission | 512 MiB layer-aware |
| --- | ---: | ---: |
| Operation acquires / releases | 80,328 / 80,328 | 73,684 / 73,684 |
| Logical source bytes | 181.21 GB | 162.89 GB |
| Logical source-byte difference | baseline | -10.11% |
| Aggregate tokens/s, c1 | 6.34 | 3.08 (-51.5%) |
| Aggregate tokens/s, c2 | 8.06 | 3.12 (-61.3%) |
| Aggregate tokens/s, c4 | 3.25 | 2.51 (-22.7%) |
| Aggregate tokens/s, c8 | 4.49 | 3.72 (-17.3%) |
| Loads / evictions | 0 / 0 | 68,196 / 67,902 |
| Provider wait | 55.7 ms | 28,287.1 ms |

The unequal operation totals show that llama.cpp formed different dynamic
batches as request timing changed. Consequently, 10.11% is an observed
whole-run difference, not a clean estimate of cache reuse. A future paired
test must replay a fixed routing schedule or record source bytes per identical
operation key. Nevertheless, the key implementation result is unambiguous:
the old control made 457,788 worker acquires; the new control made 80,328
operation acquires. Provider wait fell from 990.6 ms to 55.7 ms despite this
rerun doing a different amount of routed work.

The fresh cache/no-cache oracle used identical binary, prompt, placement, and
three greedy decode steps. Both 262,144-element output vectors have SHA-256
`836cb407bafe5aa39d8adc0363e6af5bfe9eede860af47b033b72120f55fab1`.

## Measurements

Both servers used the same Gemma-4 26B-A4B Q5_K_M artifact, 128-token prompts,
16 generated tokens, eight server slots, and request order c4, c1, c2, c8.

| Metric | No admission | 512 MiB layer-aware |
| --- | ---: | ---: |
| Correct executable answers | 14 / 15 | 15 / 15 |
| Logical backing-source bytes | 172.08 GB | 153.25 GB |
| Source-byte reduction | baseline | 10.94% |
| Aggregate tokens/s, c1 | 1.73 | 1.01 |
| Aggregate tokens/s, c2 | 1.85 | 1.28 |
| Aggregate tokens/s, c4 | 2.51 | 1.60 |
| Aggregate tokens/s, c8 | 4.04 | 2.18 |
| Cache loads / evictions | 0 / 0 | 63,180 / 62,864 |
| Peak owned cache | 0 | 536,752,128 bytes |

The cache reduced source traffic by 18.83 GB, but its old per-worker copy and
lock path reduced throughput by roughly 40% to 45%. This table is retained as
historical evidence; the operation-shared rerun above supersedes its callback
architecture and shows that cache copying and churn still need redesign.

The 15/15 versus 14/15 difference must not be interpreted as improved quality.
Only one run was made and batching can vary generation. Earlier deterministic
oracle testing remains the correctness evidence: at 512 MiB the layer-aware
provider saved 6.52% of source bytes and produced bit-identical logits.

## What changed technically

The provider now lives for the lifetime of the pinned llama.cpp server and is
enabled only with `GGML_MMID_CACHE_MIB`. `GGML_MMID_CACHE_POLICY=layer` makes an
incoming expert prefer an idle victim from the same layer, retaining a small
cross-layer working set. The build copies the reviewed provider header into the
isolated llama.cpp source only while applying and compiling the experiment
patch; the pinned upstream checkout is restored cleanly afterward.

Telemetry now reports one acquire per expert operation. The historical pilot
predates that architecture, so its checked-in worker hits cannot be treated as
cross-request cache hits. The operation-shared rerun reports 606 genuine
operation hits, only 2.03% of candidate operations.

## Blocker and next experiment

The RTX 5060 Ti initialized during both deterministic oracle runs, then
`nvidia-smi` again lost driver communication. The planned repeated
GPU-placement, VRAM, physical-I/O, RSS/PSS, and energy campaign remains blocked
on stable driver access.

Next, replace synchronous copy-on-admission with an operation-keyed shared
loader/prefetch path and test a fixed routing replay before another live-server
campaign. Once GPU access is stable, run alternating no-admission, LRU, and
layer-aware trials at c1/c2/c4/c8 with at least three repetitions. Success
means correct concurrent work with materially fewer backing-tier expert loads
on hardware that cannot hold the model in VRAM, not beating ordinary llama.cpp
for one user.

Data: [`gemma4-concurrent-native-cache-pilot.json`](data/gemma4-concurrent-native-cache-pilot.json).
