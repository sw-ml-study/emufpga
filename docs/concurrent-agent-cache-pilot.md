# Concurrent-agent native cache pilot

## Concise conclusion

The first long-lived-server pilot shows the mechanism can share copied expert
bytes across concurrent requests while preserving bounded ownership. A 512 MiB
layer-aware cache reduced logical backing-source demand by 10.94% across the
same c4, c1, c2, and c8 request sequence. It was also substantially slower than
the no-admission control. This is promising capacity/traffic evidence and a
warning that the current copy-and-global-mutex implementation is not yet useful
for production.

This is a single CPU-only pilot, not a validated project claim.

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

The cache reduced source traffic by 18.83 GB, but the copy and globally locked
provider reduced throughput by roughly 40% to 45%. Speed is not the project
goal, yet that penalty is too large to call the mechanism good enough. The high
eviction count also says this policy is still churning.

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

Telemetry now distinguishes expert operations from worker callbacks. This
pilot predates that telemetry revision, so its checked-in worker hits cannot be
treated as cross-request cache hits. A rebuilt rerun must report both.

## Blocker and next experiment

The RTX 5060 Ti appeared once through `nvidia-smi`, then CUDA initialization and
subsequent `nvidia-smi` calls failed. The planned GPU-placement, VRAM, physical
I/O, RSS/PSS, energy, and repeated paired campaign therefore remains blocked on
stable driver access.

Once restored, rebuild and run alternating no-admission, LRU, and layer-aware
trials at c1/c2/c4/c8 with at least three repetitions. The next implementation
optimization should load once per expert operation and remove the per-worker
global mutex. Success means correct concurrent work with materially fewer
backing-tier expert loads on hardware that cannot hold the model in VRAM, not
beating ordinary llama.cpp for one user.

Data: [`gemma4-concurrent-native-cache-pilot.json`](data/gemma4-concurrent-native-cache-pilot.json).
