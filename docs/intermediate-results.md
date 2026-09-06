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

## Can expert subsets make a model larger than RAM plus VRAM run?

Yes, provided the non-weight working set still fits. A bounded implementation
does not need every expert resident simultaneously. At each MoE layer it can:

1. compute routing for a batch of token activations;
2. group those activations by selected expert;
3. load one expert, or a small expert subset, from SSD;
4. apply it to every activation that selected it;
5. accumulate each token's weighted result;
6. release those expert weights and load the next subset; and
7. continue once all selected experts for the layer are complete.

The model file can therefore be larger than RAM plus VRAM because SSD is the
backing store and only a bounded expert window is resident. This does not remove
every capacity constraint. Shared dense layers, routing state, execution
buffers, activations, and the KV caches for all active requests must fit in RAM
or VRAM. Tiering KV cache to storage is possible in principle but would add a
different, latency-sensitive stream and is likely much more expensive than
streaming immutable expert weights.

### Is this different from llama.cpp using SSD-backed mmap?

Yes, although both ultimately may read model bytes from SSD. llama.cpp normally
memory-maps a GGUF file. Linux faults file-backed pages into its page cache on
demand and may evict them under memory pressure. If those pages are needed
again, they are read again. This is demand paging rather than an explicitly
scheduled expert stream. Anonymous swap is a separate mechanism and would be a
particularly undesirable fallback for runtime state.

The proposed batching policy knows the selected experts and can deliberately
order the work. It may therefore:

- issue larger, ordered reads instead of many reactive page faults;
- load an expert once and apply it to several request activations;
- prefetch the next expert while computing the current one;
- explicitly limit the expert working set; and
- avoid retaining cold experts merely because the OS has spare page cache.

In simplified terms:

```text
uncoordinated demand ~= sum of expert fetches made by each request
coordinated demand   ~= unique expert bytes needed by the whole batch
fetch reuse          ~= expert assignments / unique experts fetched
```

For example, if eight requests make 64 expert assignments at a layer but those
assignments cover 32 unique experts, a perfectly coordinated implementation
gets about two expert uses per fetch. This example explains the mechanism; it
is not a claim that every layer or real workload achieves that overlap.

### Will coordinated streaming beat ordinary SSD paging?

For one request, probably not. mmap and the Linux page cache are mature, and
the current reclaimed implementation is already slower than resident offload.

At useful concurrency it may beat uncoordinated demand paging if routing
overlap is high, work is grouped by expert, reads remain ordered, prefetch
overlaps computation, and the compute engines keep up with storage. It may lose
if routing overlap is low, batch formation adds excessive delay, the same
experts are reloaded frequently, or SSD bandwidth and latency dominate.

The honest status is:

- execution with model weights larger than RAM plus VRAM is architecturally
  plausible;
- better performance than uncontrolled SSD paging is plausible with batching;
- neither claim has yet been demonstrated end to end on this model; and
- current measurements show reduced RAM residency but worse latency.

The cold-cache qualification must measure physical SSD bytes for resident and
reclaimed policies. A later bounded-batching test must compare ordinary mmap
paging with coordinated expert loading under a memory limit where conventional
residency cannot succeed. The intended payoff is not necessarily a faster
single request. It is enabling an otherwise impossible model, then amortizing
weight reads across enough concurrent agents to provide an acceptable aggregate
work rate.

### First physical-I/O pilot: an important negative control

The pinned GGUF currently resides on one HGST spinning disk. A per-file
`POSIX_FADV_DONTNEED` probe verified zero resident pages before each cold
startup; global cache eviction was not used.

An ordinary llama.cpp build read the entire 19.32 GB GGUF during cold startup,
took 83.8 seconds to become ready, and made every model page resident. Disabling
the inference warmup did not change that behavior. It is therefore a useful
conventional cold-loader control, but cannot enable a model larger than host
memory.

The patched lazy-expert build changed the split materially:

| One cold request | Lazy, retained pages | Lazy, reclaimed pages |
| --- | ---: | ---: |
| Startup physical read | 3.83 GB | 3.83 GB |
| Request physical read | 8.12 GB | 8.12 GB |
| Request device reads | 1,982,353 | 1,982,353 |
| Average request read | 4,096 bytes | 4,096 bytes |
| Peak process RSS | 11.93 GiB | 10.44 GiB |
| Request inference time | 155.2 s | 172.0 s |
| Executable tests | 1/1 passed | 1/1 passed |

This is encouraging capacity evidence: lazy expert marking avoided reading
15.49 GB during startup and kept peak process RSS far below the full model file.
It is also a poor physical streaming result. Nearly two million 4 KiB reads are
demand paging, not sequential streaming. Reclamation reduced process RSS by
about 1.49 GiB but did not reduce physical bytes in this unconstrained pilot and
added latency. `MADV_DONTNEED` removed process mappings while the corresponding
clean pages could remain in the global page cache.

The HDD result must therefore be treated as the baseline that a purpose-built
stream must beat, not as evidence that spinning disks cannot work.
Machine-readable pilot records are checked in for the
[lazy retained](data/gemma4-q5km-cold-hdd-lazy-pilot.json) and
[lazy reclaimed](data/gemma4-q5km-cold-hdd-reclaimed-pilot.json) policies.

### How parallel SAS HDD and SSD tiers could help

A mixed SAS system permits several useful layouts:

1. Keep the canonical GGUF or archival weights on HDD and store a complete
   repacked execution-order `.spm` stream on SSD. This uses extra SSD capacity
   but gives the cleanest large-read path.
2. Keep all weights on HDD and prepopulate an SSD cache with shared weights and
   frequently selected experts. Cold experts remain on HDD. Prepopulation
   avoids unpredictable runtime cache-write traffic.
3. Place independent layer or expert stream shards across several HDDs in JBOD
   form. Issue large sequential reads from each drive into NUMA-local bounded
   buffers and give each stream to a corresponding CPU-core group.
4. Use SSD as a staging ring between parallel HDD producers and compute
   consumers when their instantaneous rates differ. This adds writes and must
   justify them through better sustained utilization.

The first two are easiest to implement and interpret. Shared/dense weights and
the hottest experts belong on SSD; immutable cold experts are the best HDD
candidates. KV cache and live activations should remain in RAM or VRAM because
they are mutable, latency-sensitive, and repeatedly accessed.

For HDDs, physical order is part of the algorithm. Expert data should be laid
out in the order consumed, reads should be large and asynchronous, and buffers
should be reused. Within one MoE layer, different selected experts can be read
from different drives and processed by different core groups. Across layers,
the activation dependency remains sequential.

RAID striping can provide aggregate bandwidth for large reads, while explicit
JBOD placement provides clearer attribution and scheduling. Both should be
tested; parity RAID is less attractive for a read-only derived stream because
the source weights are reproducible and parity adds complexity without fixing
the central access-pattern problem.

A useful tiered-storage scorecard must report:

- physical bytes and I/O operations from each drive;
- average and percentile read size, queue depth, and sequential bandwidth;
- SSD cache hit rate and cache-fill writes;
- expert assignments, unique experts, and uses per fetched expert;
- bounded RAM/VRAM residency and NUMA placement;
- task correctness, aggregate completions, and latency; and
- drive power, SSD endurance implications, and results per drive-hour or kWh.

### Relationship to an ML-first operating system

The same mechanism could become an operating-system resource manager rather
than remain private to one inference process. A conventional OS sees files,
pages, processes, and devices but does not know that a byte range is an
immutable expert tensor or that several requests will shortly need the same
range.

An ML-first OS could expose first-class resources such as:

- content-addressed model and quantization objects;
- tensor, layer, and expert ranges with declared execution order;
- immutable weight streams and bounded residency contracts;
- mutable per-request KV caches and activation buffers;
- routing batches that can be grouped by expert;
- HDD, SSD, PMEM, RAM, VRAM, GPU, CPU, and FPGA placement capabilities; and
- byte, latency, energy, endurance, and correctness accounting.

Instead of allowing each agent process to fault the same expert independently,
the OS could accept routed activation work, wait within a bounded latency
window, and dispatch all compatible activations when that expert stream passes
through memory. This is data-centric scheduling: move small activation work to
the available immutable weight stream rather than repeatedly moving large
weights to unrelated process schedules.

The separation of resource types matters. Weight streams are immutable,
reconstructible, and friendly to sequential storage. KV caches are mutable,
request-specific, and latency-sensitive. Activations are smaller and movable
but have layer dependencies. Treating all three as ordinary interchangeable
virtual-memory pages loses information needed for good placement.

A practical Linux prototype should come before a new kernel. A privileged or
user-space model resource service can use existing files, `io_uring`, cgroups,
NUMA policy, GPU APIs, and shared-memory queues while presenting the proposed
OS contract. That lets the project measure whether semantic scheduling creates
value before making kernel or ML-first-OS changes.

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
5. Repack the same selected-expert workload into a large-read sequential stream
   and compare it with the measured 4 KiB mmap-fault baseline on one HDD, then
   on parallel SAS HDDs with an SSD hot tier.

Detailed source data and analysis are in
[the Gemma-4 serial-expert report](gemma4-serial-experts.md),
[the executable comparison](data/gemma4-q5km-executable-code.json), and
[the fixed-output throughput data](data/gemma4-q5km-bounded-end-to-end.json).
