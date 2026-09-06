# Sequential expert streams on HDD and NVMe

## Conclusion

The physical mechanism works: a deterministic 169,688,672-byte Gemma-4
Q5_K_M layer-0 expert slice was consumed from an HDD in execution order with
a 2 MiB bounded double buffer. A cold pass used 46 device reads, or about
3.69 MB per read, rather than the approximately 4 KiB faults observed during
ordinary mmap inference. All 56 replays produced the same byte digest.

This is not yet evidence that serial storage improves complete inference.
The replay consumes the exact artifact bytes and performs deterministic byte
work; activation equivalence for this artifact was established separately by
the existing direct-versus-stream Rust oracle. Full-model scheduling, router
latency, expert computation, KV traffic, request completion rate, and coding
quality are outside this measurement.

## Measured results

Seven runs were made for each combination. Values below are medians. `cold`
means the exact artifact was evicted before the run; `warm` means it was
preloaded into the Linux page cache.

| storage | cache | reader | elapsed | payload rate | device reads | average device read |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HDD | cold | synchronous | 931.9 ms | 173.6 MiB/s | 46 | 3.69 MB |
| HDD | cold | double buffer | 925.5 ms | 174.8 MiB/s | 46 | 3.69 MB |
| HDD | warm | synchronous | 205.3 ms | 788.1 MiB/s | 0 | cached |
| HDD | warm | double buffer | 184.1 ms | 879.1 MiB/s | 0 | cached |
| NVMe | cold | synchronous | 286.0 ms | 565.9 MiB/s | 324 | 0.52 MB |
| NVMe | cold | double buffer | 184.8 ms | 875.6 MiB/s | 324 | 0.52 MB |
| NVMe | warm | synchronous | 203.7 ms | 794.6 MiB/s | 0 | cached |
| NVMe | warm | double buffer | 184.5 ms | 877.3 MiB/s | 0 | cached |

Cold HDD prefetch changes little because the disk is the bottleneck. On the
NVMe control it overlaps reads with the digest consumer and reaches roughly
the warm-reader ceiling. These timings must not be compared directly with a
complete llama.cpp request: the work and byte scopes differ.

## What was held constant

- Both copies have SHA-256
  `453cc38f761e0bb8f6af101081aa969b3938a8d0d7a220a2408131a8dbf8cbfd`.
- Every replay consumed 169,688,672 bytes and returned FNV-1a digest
  `3a7affa3b6a16627`.
- The reader uses `WeightStream` forward reads, never seeks, and bounds its
  two buffers to 2,097,152 bytes total.
- The HDD source was `/dev/sdb1`; the complete SSD control was NVMe storage.
- Process I/O counters establish attributable bytes. Whole-device counters
  are upper bounds because they can include unrelated host activity.

## What is still missing

This host has one suitable HDD, so independent parallel-SAS scaling was not
measured. The replay also has no expert-range placement manifest, so calling
the full NVMe copy an "SSD hot-expert cache" would be misleading. Queue-depth
histograms, per-run read-size distributions, wall power, NUMA comparisons,
cache writes, endurance, full inference latency, and completed coding tasks
remain unmeasured.

The next relevant steps are:

1. On a multi-SAS host, shard execution-order expert ranges across independent
   drives and measure aggregate completions, device queue depth, power, and
   NUMA placement rather than extrapolating from one disk.
2. Build the workload-aware expert profile in step 025, emit an expert-range
   placement manifest, put only ranked hot ranges on SSD, and record real hit,
   miss, promotion, and write-amplification data.
3. Replay identical routes and arithmetic through mmap, HDD stream, hybrid,
   and SSD control before claiming an inference benefit.

The checked-in aggregate data is
[`data/gemma4-stream-tier-analysis.json`](data/gemma4-stream-tier-analysis.json).
Reproduce it with `scripts/measure-gemma4-stream-tiers`, then
`scripts/analyze-gemma4-stream-tiers.mjs`.
