# Cold-cache and physical-I/O qualification

## Why the current counter is zero

The Gemma experiments access expert weights through `mmap`. Linux satisfied the
measured faults from warm page cache, so `/proc/PID/io` reported zero
`read_bytes`. That does **not** mean inference moved zero bytes: it means no
physical block reads were attributed to the server during that run.

## Safe experiment

Do not use global `drop_caches` on a shared machine. Instead:

1. Put a fresh copy or reflink-disabled copy of the pinned GGUF on an otherwise
   idle test device. Record its SHA-256 and the device/partition mapping.
2. Use a tiny `posix_fadvise(POSIX_FADV_DONTNEED)` helper on that exact file, or
   a delegated cgroup-v2 `memory.reclaim`, then verify residency with `mincore`.
3. Record device sectors read before and after from `/sys/class/block/*/stat`,
   server minor/major faults, `/proc/PID/io`, RSS/PSS, and elapsed time. Device
   counters require an idle device; otherwise use cgroup-aware eBPF block-I/O
   attribution when permissions allow.
4. Run resident and reclaimed policies in alternating order for at least three
   cold trials and three warm trials at concurrency 1/2/4/8. Keep model SHA,
   prompts, seed, output cap, llama.cpp revision, and GPU placement identical.
5. Report physical bytes per generated token and per passing task separately
   from logical selected-expert bytes. Include whole-system wall energy later.

## Decision

Success is not "faster." The storage path succeeds if the oversized model still
passes the same executable tasks within VRAM while physical reads and SSD wear
remain tolerable for the intended service duration. A cold path that thrashes
storage but runs correctly is a **mixed** capacity result. A path that corrupts
answers or exceeds VRAM fails. Warm-cache results must never be relabeled as SSD
streaming measurements.

## Spinning-disk and mixed SAS follow-up

The first cold pilot showed that mmap demand paging issued 1,982,353 reads of
4,096 bytes during one 62-token response. That is a deliberately unfavorable
baseline for a disk whose strength is sustained sequential transfer.

Do not use that result to reject HDD storage. Compare it with a physical `.spm`
layout that is already ordered for execution and read through bounded
asynchronous buffers. Test these tiers separately:

1. one HDD containing the execution-order stream;
2. independent stream shards on parallel SAS HDDs;
3. HDD backing weights plus a prepopulated SSD hot-expert cache; and
4. a complete SSD execution-order stream as the upper storage-tier control.

Keep shared weights and mutable KV/activation state in SSD or RAM/VRAM as
appropriate; HDD is primarily for immutable expert streams. Attribute counters
per physical device. Report average read size and sequential bandwidth in
addition to bytes, because replacing millions of 4 KiB faults with large reads
is the mechanism under test.

## Completed c1/c2/c4/c8 qualification

The matrix completed on 2026-09-06: three fresh-server repetitions of both
policies, cold and warm, at concurrency 1/2/4/8. All 180 executable Rust task
evaluations passed. Per-file eviction verified zero resident model pages before
cold starts; warm phases attributed zero physical reads. Global caches were
never flushed.

| Requests | Policy | Seconds | Physical GB | GB/task | Tasks/hour | Peak RSS GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | resident | 155.3 | 8.12 | 8.12 | 23.2 | 11.65 |
| 1 | reclaimed | 170.2 | 8.12 | 8.12 | 21.2 | 10.18 |
| 2 | resident | 198.3 | 9.19 | 4.59 | 36.3 | 12.71 |
| 2 | reclaimed | 221.6 | 9.19 | 4.59 | 32.5 | 10.13 |
| 4 | resident | 235.4 | 10.14 | 2.54 | 61.2 | 13.67 |
| 4 | reclaimed | 320.7 | 10.16 | 2.54 | 44.9 | 10.89 |
| 8 | resident | 262.2 | 10.70 | 1.34 | 109.8 | 14.28 |
| 8 | reclaimed | 308.9 | 10.70 | 1.34 | 93.2 | 12.59 |

Eight tasks required only 1.32 times c1's physical bytes. Unconditional
reclamation saved 1.47--2.79 GiB peak RSS but lowered completion rate without
lowering physical traffic. This motivates a budgeted hot-expert cache. Reads
still averaged about 4,096 bytes, so this remains the negative mmap control,
not evidence for the proposed large sequential stream.
