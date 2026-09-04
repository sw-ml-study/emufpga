# Xeon architecture considerations for streamed MoE experts

This note ranks the older Xeon systems available to this project for the
specific job of serving multiple requests from an MoE model too large for GPU
memory. It is not a generic CPU ranking. The relevant operation is repeatedly
reading one expert's packed weights, decoding them, and applying them to every
queued activation routed to that expert.

## Short answer

The likely order of interest is:

1. **ProLiant Gen10 Plus / 3rd-generation Xeon Scalable**
2. **ProLiant Gen10 / 2nd-generation Cascade Lake Xeon Scalable**
3. **ProLiant Gen9 or Dell T7910 / Xeon E5-2600 v4**
4. **ProLiant Gen10 / 1st-generation Skylake Xeon Scalable**
5. **ProLiant Gen8 / Xeon E5-2600 v1 or v2**

The order is driven primarily by populated memory channels, NUMA locality,
sustained memory bandwidth, vector instructions, and all-core clock. Maximum
core count is not automatically best.

The exact DIMM population can outweigh the CPU model. An eight-channel CPU
with only four channels populated behaves like a four-channel memory system.

## Do not permanently assign 32 experts to 32 cores

A permanent expert/core mapping is simple but usually wastes cores when routes
are skewed. It can also make some requests wait for one overloaded expert while
other expert cores are idle.

Use an expert-centric worker pool instead:

```text
multiple requests
       |
       v
router results
       |
       v
per-expert token queues
       |
       v
N workers per NUMA node
       |
       v
each worker scans one expert for every queued token
```

The weights remain sequential within an expert. A worker takes the next
populated expert queue rather than owning one expert forever. Increase worker
count only until local memory bandwidth saturates. Test 1, 2, 4, 8, 12, 16,
24, and 32 physical workers per socket; start without SMT siblings.

For dual-socket systems:

- allocate each expert on the socket that will process it;
- pin workers and memory to the same NUMA node;
- partition expert ownership between sockets;
- avoid remote memory reads in the hot path;
- exchange only route metadata, activations, and combined expert outputs.

A global work queue whose weights were first-touched on the wrong socket can
make a newer server appear slower than an older one.

## What determines performance

For batch-one or small-batch expert GEMV, the likely priority is:

1. sustained local memory bandwidth;
2. number and population of memory channels;
3. NUMA locality;
4. packed-weight decoding efficiency;
5. useful SIMD shuffle, mask, and dot-product instructions;
6. sustained all-core frequency;
7. core count.

Larger batches reuse each expert scan across several routed activations. That
increases arithmetic per byte, making SIMD execution and clock frequency more
important. There may consequently be two winners: a many-channel system for
small-batch streaming and a higher-frequency system for larger expert batches.

## Gen10 Plus: likely best overall

Interesting Ice Lake-era candidates include:

- Gold 6346: 16 relatively high-frequency cores;
- Gold 6354: 18 high-frequency cores;
- Gold 6342: 24 cores;
- Gold 6348: 28 cores;
- Gold 6338: 32 lower-frequency cores.

A dual 6342 or 6348 is a strong aggregate-throughput candidate. A 6346 or 6354
may be better when individual expert latency matters more. Third-generation
Xeon Scalable platforms provide eight memory channels per socket, DDR4-3200,
and PCIe 4.0, which also improves CPU-to-GPU expert staging. See
[Intel's Ice Lake feature summary](https://www.intel.com/content/www/us/en/support/articles/000099418/processors/intel-xeon-processors.html).

Do not infer every instruction from the marketing generation. Inspect each
machine's actual flags:

```sh
lscpu | rg 'Model name|Flags'
```

Relevant flags include `avx512_vnni`, `avx512_bf16`, `avx512_vbmi`,
`avx512_vbmi2`, `avx512_bitalg`, and `avx512_vpopcntdq`. Cooper Lake introduced
native AVX-512 BF16, while Gen10 Plus systems are commonly Ice Lake. Confirm
the installed CPU rather than assuming BF16 support from “third generation.”
[Intel's AVX-512 generation table](https://www.intel.com/content/www/us/en/support/articles/000058341/processors/intel-xeon-processors.html)
distinguishes VNNI and BF16 support.

## Gen10 Cascade Lake: important for PMEM 100

Interesting choices include:

- Gold 6254 or 6246 for high per-core performance;
- Gold 6248/6248R for balance;
- Gold 6242/6242R;
- Gold 6230/6230R for more throughput-oriented configurations;
- Platinum 82xx when already installed or required for a large PMEM setup.

Cascade Lake supplies six memory channels per socket, AVX-512 VNNI, and PMEM
100 support. VNNI directly helps INT8 dot products. It helps 1.58-bit weights
only after they are unpacked to a compatible representation or when the
ternary kernel deliberately maps its operations to those instructions. See
[Intel's second-generation Xeon Scalable brief](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/2nd-gen-xeon-scalable-processors-brief.html).

First-generation Skylake Scalable CPUs remain useful AVX-512 baselines but do
not have AVX-512 VNNI.

## Gen9 and Dell T7910: useful upcycling candidates

Prefer E5-2600 v4 over v3 when cost and compatibility are similar. Candidate
sweet spots include:

- E5-2697A v4: 16-core balance;
- E5-2690 v4: 14 cores and respectable clock;
- E5-2680 v4: common and economical;
- E5-2667 v4: fewer, faster cores;
- E5-2699 v4: many cores, but all share four memory channels.

These systems have AVX2/FMA and four DDR4 channels per socket. Intel's
[E5-2600 v4 memory validation document](https://www.intel.com/content/dam/www/public/us/en/documents/platform-memory/ddr4-lrdimm-xeon-e5-v4-validation-results.pdf)
documents four channels and DDR4-2400 configurations.

For sequential GEMV, two 22-core E5-2699 v4 CPUs may not scale proportionally
over two 14- or 16-core CPUs because memory bandwidth saturates before all
cores become useful. The larger CPUs are still valuable when expert decoding
becomes compute-bound or when many independent non-memory-bound tasks coexist.

## Gen8: control and low-cost capacity tier

E5-2600 v2 systems have AVX but not AVX2 or VNNI and use DDR3. Interesting
installed parts include E5-2667 v2 for per-worker latency and E5-2690 v2 or
E5-2697 v2 for aggregate work. They are useful as a negative/control baseline
and may remain economically useful when chassis, memory, and power are already
available, but they are unlikely to lead in performance per watt.

## Why 1.58-bit weights may change the winner

“1.58 bit” normally describes ternary information content, not necessarily the
physical file width. Implementations may use two bits per weight, bit planes,
several trits packed per byte, block metadata, and scales.

Compression reduces bytes but adds decoding work. That can move an expert from
memory-bound to decode- or compute-bound:

- scalar unpacking favors frequency and may erase the compression benefit;
- AVX2 lookup/shuffle decoding can make Gen9 viable;
- AVX-512 byte, bit, mask, and population-count operations favor newer CPUs;
- unpack-to-INT8 followed by VNNI favors Cascade Lake and later;
- a true ternary add/subtract kernel favors efficient masks and accumulators.

Every quantization experiment should report:

```text
physical bits per weight including metadata
packed read GB/s
decoded weights/s
useful ternary or MAC operations/s
tokens/s at each batch and concurrency
CPU package joules/token where RAPL is available
accuracy or perplexity change
```

A smaller artifact is not a successful quant if unpacking consumes all saved
time or model quality falls unacceptably.

## PMEM 100 and PMEM 200

These systems are first-class experimental targets for upcycling. PMEM can
hold immutable experts close to a CPU at capacities larger than conventional
DRAM budgets.

Test these modes separately:

- DRAM;
- PMEM App Direct through a normal filesystem;
- PMEM App Direct with DAX;
- PMEM Memory Mode;
- local NVMe.

Memory Mode uses DRAM as a cache for PMEM. A large one-pass weight stream can
pollute that cache, so it may be worse than explicit placement. App Direct/DAX
is conceptually attractive because experts are independently addressable and
can be mapped without pretending to be ordinary DRAM. Intel explains the
[Memory Mode and App Direct distinction](https://www.intel.com/content/www/us/en/support/articles/000055895/memory-and-storage/intel-optane-persistent-memory.html).

PMEM 100 is tied to compatible second-generation Xeon Scalable processors;
see [Intel's compatibility list](https://www.intel.com/content/www/us/en/support/articles/000055996/memory-and-storage/intel-optane-persistent-memory.html).
PMEM 200 plus an eight-channel Gen10 Plus host is probably the fleet's most
interesting “large expert store close to compute” configuration.

Interleave PMEM across all available channels and record whether the namespace
is filesystem, devdax, or sector mode. Results from one DIMM cannot represent a
fully populated system.

## CPU versus GPU placement

For the actual goal—multiple requests sharing a model too large for one
GPU—the likely useful hybrid is:

1. keep shared layers, KV cache, and frequently used experts in GPU memory;
2. group waiting tokens by selected expert;
3. keep long-tail experts in pinned DRAM or PMEM;
4. asynchronously stage the next expert while the GPU computes the current
   one;
5. retain a popularity-aware GPU expert cache;
6. measure PCIe transfers and synchronization separately.

Executing experts on CPU avoids transferring their weights but can require
CPU/GPU synchronization at every MoE layer. Transferring selected experts to
the GPU spends PCIe bandwidth but gains GPU arithmetic. Both designs must be
measured. The correct answer changes with quant width, expert reuse across
requests, PCIe generation, and CPU SIMD capability.

## Portable machine bake-off

Capture this inventory on every host:

```sh
lscpu
numactl --hardware
dmidecode -t processor -t memory
ndctl list -R -N -M
daxctl list
lsblk -o NAME,SIZE,ROTA,TRAN,MODEL,FSTYPE,MOUNTPOINTS
nvidia-smi --query-gpu=name,memory.total,pci.bus_id --format=csv
```

Then run the same model hash, prompts, route trace, layout, compiler settings,
and worker sweep on every machine. Required axes are:

- Q6_K, representative Q4, INT8/BF16 where supported, and 1.58-bit/ternary;
- batch 1, 2, 4, 8, 16, and 32;
- 1 through saturation physical workers per NUMA node;
- local versus remote NUMA placement;
- DRAM, PMEM filesystem, PMEM DAX, and NVMe;
- warm, cache-bypass proxy, and genuine cold-start where safely measurable;
- CPU-only expert execution versus asynchronous GPU staging;
- throughput, p50/p90/p99 latency, bytes, CPU utilization, and energy.

Expected candidates, to be tested rather than assumed:

- **Best aggregate CPU/DRAM:** dual Gen10 Plus Gold 6342/6348 class with all
  memory channels populated.
- **Best latency-oriented CPU:** Gold 6346/6354 class.
- **Best PMEM 100 platform:** dual Cascade Lake Gold 6248/6254/62xxR class.
- **Best Gen9 upcycle candidate:** dual E5-2690 v4 or E5-2697A v4.
- **Less promising:** maximum-core-count Gen8/Gen9 CPUs with partially
  populated channels.

These are hypotheses. The project should select hardware from measured
bandwidth saturation, decoded-weight throughput, end-to-end tokens/s, and
energy—not model numbers alone.
