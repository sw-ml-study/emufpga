# Workload-aware expert caching (RTX 5060 Ti)

## Goal and interim conclusion

The goal is **not** to outperform conventional `llama.cpp` offload for one
request. Single-request offload is a correctness and cost control. The target
is to let multiple concurrent coding agents share each selected expert's
weight residency, transfer, decode, and potentially FPGA/CPU matrix-processing
window instead of independently paying those costs for every request.

The success question is therefore: as useful agents increase from one to two,
four, and eight, can aggregate correct work increase while physical expert
traffic and expert setup work grow materially slower than the request count?
Per-agent latency may rise. A successful result does not require lower
single-user latency.

A calibration-ranked expert cache transfers moderately well to disjoint Rust
coding tasks, but the first 1 GiB physical retention experiment did **not**
produce a measurable throughput or disk-I/O benefit over Linux's normal file
cache. This is useful negative evidence: logical expert reuse is not the same
as avoidable physical I/O.

These results do not reject bounded expert caching. They reject the narrower
assumption that leaving selected mmap pages mapped, while applying
`MADV_DONTNEED` to the rest, will outperform the kernel page cache on this
RAM-rich host without controlled memory pressure.

It does not yet test the stronger project hypothesis: coalescing selected
experts across simultaneous requests and processing the combined activation
batch during one expert residency window. Current `llama.cpp` batching may
reuse OS-cached pages, but this experiment has not proven that it schedules
cross-request expert work as the proposed serial service would.

## Protocol

- Model: pinned Gemma-4 26B-A4B Q5_K_M GGUF (19.32 GB file).
- Placement: dense/shared work and KV cache on the RTX 5060 Ti; MoE expert
  tensors forced to CPU.
- Workload: four calibration and four disjoint held-out executable Rust coding
  tasks, 128 output-token limit, concurrency four.
- Cache ranking uses calibration routes only. Held-out routes never influence
  the static ranking.
- Physical comparison: no retained experts versus a calibration-ranked 1 GiB
  retained set, three sequential repetitions each, cold file before server
  load, server warmup enabled.

## Routing evidence

The calibration trace contained 51,018 expert accesses; held-out contained
45,723. Their top-20 layer/expert sets overlap by 75%. Held-out reuse distance
was 532 accesses at p50 and 2,401 at p90.

Trace replay projected static-cache byte-hit rates of 15.6% at 256 MiB, 28.5%
at 512 MiB, 48.2% at 1 GiB, 70.2% at 2 GiB, and 89.8% at 4 GiB. An adaptive
admit-after-two LFU/LRU simulation was weaker at every tested budget. These are
logical trace projections, not physical measurements.

## First physical comparison

| Policy | Correct | Mean c4 batch | Aggregate tok/s | Mean device reads |
|---|---:|---:|---:|---:|
| Reclaim all experts | 12/12 | 77.89 s | 2.953 | 2.438 GiB |
| Static calibration-ranked 1 GiB | 12/12 | 78.17 s | 2.942 | 2.418 GiB |

Static versus reclaim-all was +0.36% batch time, -0.36% throughput, and -0.84%
mean physical reads. None is a defensible improvement with three repetitions.
The first repetition read about 7.2 GiB in both policies; subsequent repetitions
read only 9--26 MiB. Consequently the mean and confidence interval are dominated
by cache history, not the application policy.

`MADV_DONTNEED` removes the process's mappings but does not guarantee eviction
of clean file-backed pages from Linux's page cache. A useful next experiment
must constrain available RAM or use a cache design whose capacity and I/O are
controlled explicitly. Repeatedly evicting the entire live model is not a valid
substitute because it also discards dense/shared pages that both policies need.

## Direct four-agent sharing result

The four held-out tasks were traced for three repetitions each in isolation and
compared with three c4 runs of the identical prompts, seed, model, cache policy,
and 128-token limit. All 24 evaluated outputs passed their executable tests,
and both arrangements generated 690 completion tokens in total.

| Metric | Four isolated traces | Concurrent c4 | Change |
|---|---:|---:|---:|
| Unique layer/expert touches | 171,856 | 128,329 | -25.3% |
| Logical expert traversal | 773.76 GB | 578.61 GB | -25.2% |
| Assignments per expert touch | 1.031 | 1.436 | +39.3% |
| Inference elapsed time | 104.19 s | 68.83 s | -33.9% |
| Aggregate completion tok/s | 6.62 | 10.02 | +51.4% |

Concurrent execution had 4.1% more routed assignments despite matching total
completion tokens, so normalization matters. Logical bytes per assignment fell
28.1%. The assignment difference likely reflects prompt batching, padding, or
graph-shape behavior and needs deeper attribution.

This is useful but strongly sublinear scaling: four concurrent agents did not
produce work at four times the isolated aggregate rate. Individual latency also
rose under shared CPU/GPU contention. At this stage the evidence supports
amortization, not "four agents each proceed at the speed of one."

The matched sharing curve is:

| Concurrent agents | Logical byte reduction | Aggregate tok/s | Gain vs sequential |
|---:|---:|---:|---:|
| 1 | 0% | 6.62 | 0% |
| 2 | 13.1% | 8.27 | +24.9% |
| 4 | 25.2% | 10.02 | +51.4% |

Across the three paired repetitions, c4 logical-byte reduction was 25.2% with
a 95% t-interval half-width of 7.4 percentage points. The paired throughput
gain averaged 48.5% with a much wider 36.6-point half-width. Reduced logical
expert work is therefore better established than the exact speed gain.

The two c2 task pairs differed sharply: logical traversal fell 25.7% for one
pair but only 6.7% for the other. Expert sharing therefore depends on workload
and routing overlap, not concurrency alone.

For eight agents, the matched baseline is the same eight tasks processed as
two c4 batches. All 24 outputs passed in both arrangements. C8 generated 1,526
tokens versus 1,506 for the c4 batches, so normalized comparisons are required:
c8 used 17.4% fewer logical expert bytes per routed assignment and improved
aggregate completion throughput by another 19.6%. Moving from c4 to c8 still
helps, but much less than moving from sequential execution to c4.
The paired c8 byte-per-assignment reduction was 17.5% +/- 2.9 points at 95%,
while its throughput gain was 18.0% +/- 25.6 points. The latter interval crosses
zero, so more repetitions are needed for a strong c8 speed claim.

This directly shows that the current batched `mul_mat_id` kernel already
amortizes expert traversal across agents. It does not yet show physical storage
savings, an application-owned serial scheduler, or FPGA acceleration. The next
question is how much additional sharing a scheduler can obtain by deliberately
holding compatible activations until the same expert can serve them together.

Machine-readable inputs and derived results are in
`docs/data/gemma4-expert-cache-analysis.json` and
`docs/data/gemma4-expert-cache-physical.json`; the direct-sharing result is in
`docs/data/gemma4-agent-sharing.json`.

## Next validation

1. Record per-layer expert selections with request/sequence identity and
   quantify simultaneous cross-request overlap.
2. Establish an independent-request baseline in which requests cannot share an
   expert residency or processing window.
3. Implement a shared scheduler that groups ready activations by layer/expert,
   loads or decodes that expert once, processes every waiting request, and then
   releases it.
4. Measure c1/c2/c4/c8 using useful work per agent, aggregate correct results,
   expert bytes per generated token, expert loads per result, throughput, and
   tail latency. Treat c1 as a control, not the optimization target.
5. Test bounded memory pressure or an application-owned expert buffer/cache;
   later substitute FPGA expert processing without changing scheduler semantics.
6. Preserve executable-task correctness and report physical I/O, faults,
   residency, CPU/GPU utilization, and whole-system energy where available.

## Deliberate scope exits

The online LFU/LRU policy was evaluated by trace replay but was not promoted to
a physical `madvise` experiment. The fixed policy could not be distinguished
from Linux page-cache behavior, so another policy using the same mechanism
would not answer the application-cache question. A physical adaptive cache now
requires an application-owned bounded buffer or controlled memory pressure.

C and assembly domain-shift corpora are deferred. This step first needed a
same-language workload to test the user's cross-agent sharing hypothesis, and
the four Rust tasks already show large task-mix sensitivity. Whole-system energy
was unavailable; NVIDIA board telemetry excludes CPU, RAM, and disks and is not
reported as agents per kWh.
