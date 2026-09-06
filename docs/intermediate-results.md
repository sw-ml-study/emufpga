# Intermediate results: oversized MoE inference

## Concise conclusion

The current experiment compares conventional hybrid CPU/GPU offload with an
experimental bounded-residency policy that releases selected MoE expert pages
after using them. It does **not** yet compare conventional offload with a proven
direct SSD-to-compute streaming implementation.

On this machine, conventional offload is presently the better default because
the machine has enough system RAM. The reclaimed serial-expert path is slower,
but uses less resident RAM. It becomes valuable when memory capacity determines
whether the model can run at all.

The strongest result so far is therefore:

> Gemma-4 26B-A4B Q5_K_M does not fit entirely in the RTX 5060 Ti's VRAM, but
> both conventional CPU/RAM offload and the bounded-residency expert policy run
> it successfully. The bounded policy trades approximately 1.6--3.2 GiB of
> resident RAM for approximately 1.5--2.1 times the inference latency in the
> measured executable workload.

We have **not** yet shown that this model cannot run conventionally on the whole
machine but can run with serial processing. A controlled memory-limit test is
needed to establish that claim.

## What is being compared?

All successful comparisons use the identical pinned Gemma-4 26B-A4B Q5_K_M
model, llama.cpp revision, GPU-layer request, prompts, seeds, and decoding
settings.

1. **All-GPU capacity baseline:** loading the complete model into GPU memory
   requests about 18,409 MiB and fails against approximately 16,311 MiB of
   usable VRAM. This is a capacity-failure baseline, not a throughput baseline.
2. **Conventional resident offload control:** suitable state is placed on the
   GPU while CPU expert pages are allowed to remain resident in system RAM.
3. **Bounded-residency candidate:** placement is otherwise identical, but the
   selected expert ranges receive `MADV_RANDOM` before use and
   `MADV_DONTNEED` after use so Linux may reclaim their file-backed pages.

The third case demonstrates selected-expert computation with a reduced
resident working set. Because Linux can still satisfy accesses from its page
cache, it is not yet evidence of bounded physical SSD traffic. The active
cold-cache experiment is intended to measure that distinction.

## Memory, latency, and correctness results

Two repetitions of eight small dependency-free Rust function tasks were run at
concurrency 1, 2, 4, and 8. Generated programs were compiled and tested inside
restricted, network-disabled containers. The hidden deterministic tests were
not included in the prompts.

| Concurrent requests | Resident RSS | Reclaimed RSS | Resident mean inference | Reclaimed mean inference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12,118 MiB | 10,556 MiB | 10.75 s | 16.52 s |
| 2 | 13,264 MiB | 10,836 MiB | 10.59 s | 22.47 s |
| 4 | 14,376 MiB | 11,163 MiB | 16.92 s | 34.90 s |
| 8 | 15,165 MiB | 12,962 MiB | 27.57 s | 52.20 s |

Peak VRAM was approximately 4,150--4,152 MiB under both policies. Both policies
compiled and passed all 30 programs. There were zero paired pass/fail
disagreements and 28 of 30 responses were text-identical. The other two were
different correct palindrome implementations.

This is evidence that reclamation did not change the answers in this bounded
corpus. It is not proof of repository-scale coding-agent quality.

## Throughput and diminishing returns

A separate fixed-output run used 128 prompt tokens and exactly 16 generated
tokens per request. This avoids confusing throughput with variable response
lengths:

| Concurrent requests | Aggregate tokens/s | Tokens/s per request | Aggregate gain vs. one request |
| ---: | ---: | ---: | ---: |
| 1 | 1.80 | 1.80 | 1.00x |
| 2 | 2.50 | 1.25 | 1.39x |
| 4 | 3.66 | 0.92 | 2.04x |
| 8 | 4.80 | 0.60 | 2.67x |

Concurrency increases total output, but much less than linearly. At eight
requests the server produces about 2.67 times as many aggregate tokens as at
one request, while each request receives only about one third of the
single-request token rate. Median time to first token rises from about 6.8
seconds at concurrency one to about 20.1 seconds at concurrency eight.

Consequently, it is inaccurate to say that eight agents proceed at nearly the
same rate as one. They can produce more work collectively, but each agent waits
substantially longer. Eight is the largest concurrency measured; no conclusion
about scaling to 16 or 32 is currently justified.

MoE sharing can amortize expert fetches when requests select overlapping
experts. That benefit eventually encounters diminishing returns because:

- the union of selected experts grows as more independent requests are batched;
- expert matrix computation still grows with the number of token activations;
- attention, KV-cache work, and non-expert layers remain request-specific;
- larger batches require more KV cache and activation memory; and
- waiting to form or synchronize a batch increases latency.

## Are the agent tasks suitable?

They are suitable as small semantic regression probes: every task has a fixed
Rust signature and hidden assertions, so a response must compile and behave
correctly rather than merely contain an expected phrase.

They are not sufficient to validate coding agents. The tasks are only `clamp`,
`gcd`, `sum_even`, `palindrome`, `count_byte`, `sorted`, `factorial`, and
`binary_search`. They do not test repository exploration, long context,
multi-file edits, tool selection, test diagnosis, patch application, or
multi-step recovery.

A defensible multi-agent result requires repository-level tasks with isolated
worktrees, deterministic acceptance tests, matched prompts and context, and
quality scoring independent of throughput. Throughput alone cannot establish
that agents produce useful or equally reliable work.

## Pros and cons

| Approach | Advantages | Disadvantages |
| --- | --- | --- |
| All-GPU | Usually lowest transfer overhead and best interactive latency | This model cannot allocate in the current 16 GiB GPU |
| Conventional CPU/RAM offload | Faster in current measurements; mature; benefits from DRAM and page-cache reuse | Needs a larger resident system-memory working set; a sufficiently large model or small host can still fail |
| Reclaimed selected experts | Reduces resident RAM; offers a path toward models larger than RAM/VRAM; fixed weights are candidates for SSD, PMEM, or accelerator streaming | Approximately 1.5--2.1x slower here; physical storage demand is not yet known; more page faults and data movement; experimental implementation |

## Is it worthwhile?

It is worthwhile **if it crosses a capacity boundary**: for example, normal
offload fails under a 14 GiB host-memory budget while the bounded policy runs
the same model and passes the same quality tests. If both approaches already
fit, the current implementation is less efficient than conventional offload
and offers no speed advantage.

This project's intended value proposition is not faster inference. It is
running a useful model on reused hardware where normal allocation strategies
cannot run it, at a slow but acceptable service level. Success therefore needs
all of the following:

1. a conventional same-model, same-quant baseline that fails a real capacity
   limit;
2. a bounded policy that stays within that limit;
3. equivalent task-quality results;
4. measured physical storage traffic and acceptable device endurance;
5. a documented service envelope for latency and concurrent agents.

## Would more or better CPU cores, or an accelerator, help?

Probably, but this is a hypothesis to measure rather than a demonstrated
result. The memory-capacity benefit comes from keeping only a bounded expert
working set resident; changing CPUs does not itself improve that bound. Better
compute can instead reduce the latency penalty paid for decoding and processing
each selected expert.

The most promising CPU system is not necessarily the one with the largest core
count. Serial expert processing needs a balanced path:

`storage -> page cache or DRAM -> memory channels -> dequantization -> matrix computation`

More cores help only while they receive weight bytes and activation work fast
enough. Once storage or memory bandwidth is saturated, additional cores contend
for the same bytes and provide diminishing or negative returns. Newer cores can
still help through stronger vector instructions, better cache behavior, faster
dequantization, and higher per-core throughput. More memory channels, NUMA-local
placement, and sufficient SSD or PMEM bandwidth may matter as much as core
count.

There are two different forms of parallelism worth testing:

- assign independent selected experts or requests to different CPU cores, then
  combine their activation contributions; and
- broadcast one fetched expert-weight stream across a batch of request
  activations, amortizing the weight read while keeping request state separate.

The second is closer to the project's central value proposition. Merely
splitting experts across cores may improve latency, but it does not automatically
reduce bytes fetched. Batching several activations against the same streamed
weights can improve useful work per byte, provided routing overlap is high
enough and batch waiting does not become excessive.

An FPGA or other offload engine bears fruit only if it processes the stream near
the data rate and avoids copying the same weights through several constrained
links. A small FPGA attached through a slow interface is unlikely to beat a
modern Xeon for this model. It can still validate ordering, buffering, and
fixed-function decode/MAC behavior. A useful accelerator experiment needs to
report end-to-end bandwidth, compute utilization, transfer energy, latency, and
correctness--not just theoretical multiply-accumulate throughput.

Expected outcomes are therefore:

- **better CPU cores:** likely lower latency and better tokens/s;
- **more cores with adequate memory bandwidth:** likely higher concurrent
  throughput, with diminishing returns at the bandwidth limit;
- **more cores without more bandwidth:** little benefit or regression;
- **faster SSD or PMEM:** useful only if physical reads, rather than compute or
  DRAM traffic, are the measured bottleneck;
- **purpose-built offload:** potentially valuable when it consumes ordered
  weights in one pass and reuses each fetched weight across multiple request
  activations, but currently unproven.

The next measurements must identify the active bottleneck before selecting a
different server or accelerator. Otherwise a faster component may optimize a
stage that is already waiting on another one.

## Next measurements

1. Finish the cold/warm cache experiment without globally dropping caches.
   Measure actual device reads, faults, bytes per generated token, RSS, PSS,
   VRAM, and latency.
2. Apply a controlled host-memory limit between the two measured working sets.
   Demonstrate conventional offload failing while the bounded policy completes
   the identical tests.
3. Replace the toy-only qualification with repository-level coding tasks and
   compare one, two, four, and eight agents on both quality and completion rate.
4. Only after those pass, increase server slots and context budget to measure
   concurrency 12 and 16. Do not extrapolate beyond eight from current data.

Detailed source data and analysis are in
[the Gemma-4 serial-expert report](gemma4-serial-experts.md),
[the executable comparison](data/gemma4-q5km-executable-code.json), and
[the fixed-output throughput data](data/gemma4-q5km-bounded-end-to-end.json).
