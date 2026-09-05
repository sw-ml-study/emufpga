# OLMoE same-quant placement baseline

## Question and experiment order

This experiment does **not** compare a large quant in VRAM with a smaller
quant in RAM. Each row uses the exact same GGUF bytes, prompt length, generated
length, and concurrency. Only placement changes:

| Quant row | All GPU | Expert tensors in CPU RAM | Ordered serial experts |
| --- | --- | --- | --- |
| Q6_K | measure | measure | adapter not implemented |
| Q2_K | measure after Q6_K | measure after Q6_K | decoder not implemented |

The middle column is a llama.cpp placement control. It keeps attention and KV
on the GPU while forcing tensors matching `.*ffn_.*_exps.weight` into CPU RAM.
It is **not** the project's ordered serial stream. It measures the alternative
that a real serial implementation must beat.

Q6_K comes first because this repository already has an independently tested
210-byte/256-weight Q6_K decoder and a Granite serial path. Q2_K comes second,
so a new decoder cannot be confused with a new placement policy.

## Pinned qualification model

- Model: AllenAI OLMoE-1B-7B-0125-Instruct, 6.92B total parameters in the
  GGUF, 64 experts, top 8, 16 layers, 4,096-token trained context.
- GGUF repository revision:
  `2ac1d27317927518c9ef7bd99f91f3f1e4ee288d`.
- Q6_K: 5,684,748,288 bytes; SHA-256
  `075a946cf693221a409054e86051b560ac688518476d2a87cc57c77a5103f224`.
- Q2_K: 2,562,762,752 bytes; SHA-256
  `56b07693cb296ba7bb8ca167d2fc2e71689cc4faa9c6a588bccaabdfb823991b`.
- Source: <https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF>.

Large artifacts live under `/disk1/tmp/olmoe-1b-7b-0125` and are not tracked.

## Fixed throughput contract

`scripts/bench-olmoe-placement` verifies the artifact hash and runs 1, 2, 4,
and 8 parallel sequences. Each sequence has 3,840 prompt tokens and 256
generated tokens, exactly filling the model's trained 4,096-token context.
Default qualification requires three runs. Raw logs, process-RSS samples, and
synchronized 200 ms GPU memory/utilization/power samples remain outside Git.

`llama-batched-bench` supplies controlled synthetic throughput, not independent
semantic tasks. It does not report TTFT, inter-token p50/p95, or correctness.
Those remain required from the server/task harness before any agents/hour or
correct results/kWh claim.

## Measured Q6_K and Q2_K placement baselines

These are medians of three runs from 2026-09-04. llama.cpp was revision
`6e6725459a892b49602b596339de4916c7c7965a`, built for CUDA SM120. Host: Xeon
W-2135 (6 cores/12 threads), 251 GiB RAM, RTX 5060 Ti 16,311 MiB. The checked-in
derived data preserve min/median/max and scoped telemetry:

- [`data/olmoe-q6-placement.json`](data/olmoe-q6-placement.json)
- [`data/olmoe-q2-placement.json`](data/olmoe-q2-placement.json)

| Quant | Placement | Parallel | Prompt tok/s | Generation tok/s aggregate | Generation tok/s/request | End-to-end seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q6_K | all GPU | 1 | 1,380.57 | 43.74 | 43.74 | 8.54 |
| Q6_K | all GPU | 2 | 1,322.14 | 25.10 | 12.55 | 26.18 |
| Q6_K | all GPU | 4 | 1,042.95 | 34.69 | 8.67 | 43.33 |
| Q6_K | all GPU | 8 | 851.36 | 47.75 | 5.97 | 78.13 |
| Q6_K | experts in CPU RAM | 1 | 145.78 | 32.78 | 32.78 | 34.56 |
| Q6_K | experts in CPU RAM | 2 | 125.08 | 44.93 | 22.46 | 71.35 |
| Q6_K | experts in CPU RAM | 4 | 103.85 | 48.37 | 12.09 | 169.07 |
| Q6_K | experts in CPU RAM | 8 | 74.20 | 44.87 | 5.61 | 459.64 |
| Q2_K | all GPU | 1 | 1,227.08 | 43.28 | 43.28 | 9.19 |
| Q2_K | all GPU | 2 | 1,198.09 | 23.86 | 11.93 | 27.87 |
| Q2_K | all GPU | 4 | 1,004.58 | 32.78 | 8.19 | 47.42 |
| Q2_K | all GPU | 8 | 795.34 | 44.69 | 5.59 | 84.45 |
| Q2_K | experts in CPU RAM | 1 | 167.55 | 50.71 | 50.71 | 28.06 |
| Q2_K | experts in CPU RAM | 2 | 150.40 | 65.46 | 32.73 | 59.47 |
| Q2_K | experts in CPU RAM | 4 | 114.98 | 71.21 | 17.80 | 148.57 |
| Q2_K | experts in CPU RAM | 8 | 93.37 | 49.73 | 6.22 | 372.23 |

Telemetry covers model loading and the complete 1/2/4/8 sweep in each run, so
it is placement-run scoped rather than attributable to one concurrency row.

| Quant | Placement | Peak process RSS median | Peak VRAM median | GPU-board energy median |
| --- | --- | ---: | ---: | ---: |
| Q6_K | all GPU | 5,804 MiB | 11,977 MiB | 5.95 kJ |
| Q6_K | experts in CPU RAM | 6,163 MiB | 6,982 MiB | 23.12 kJ |
| Q2_K | all GPU | 2,827 MiB | 9,095 MiB | 6.18 kJ |
| Q2_K | experts in CPU RAM | 3,198 MiB | 6,875 MiB | 19.07 kJ |

Q6_K CPU placement saves about 4,995 MiB peak VRAM, but the long execution
raises scoped GPU-board energy about 3.9x. Q2_K saves only about 2,220 MiB peak
VRAM. Its CPU-expert generation throughput is higher at parallelism 1, 2, and
4, but prompt processing is 7--9x slower; with the fixed 3,840-token prompt,
median end-to-end time is still 3.1--4.4x the all-GPU time. At parallelism 8,
generation throughput also becomes variable and loses its clear advantage.

This is evidence for a capacity/performance trade, not yet for the proposed
serial mechanism. It also warns against presenting generation-only throughput
as service throughput: prefill reverses the apparent Q2_K conclusion.

## What is still missing

1. Independent requests with correctness, TTFT, and inter-token percentiles.
2. Whole-system wall energy. NVIDIA power samples cover only the GPU.
3. An OLMoE adapter for the ordered Q6_K expert stream. The existing executable
   hard-codes Granite's 1,024 width, 512 FF dimension, 32 experts, and 24 layers;
   OLMoE uses 2,048 width, 1,024 FF dimension, 64 experts, and 16 layers.
4. A validated Q2_K decoder and the ordered-serial placement column for Q2_K.

Until items 1--3 exist, this is a measured placement baseline, not validation
of serial OLMoE execution.

## Apple M3 Pro 18 GB qualification role

An 18 GB unified-memory MacBook Pro is useful as a portability and efficiency
control, but it cannot reproduce the discrete-memory experiment on large12.
Its CPU and GPU share one physical memory pool. Moving an expert from Metal to
the CPU changes the execution engine and possibly the allocation policy; it
does not move the expert from VRAM into a separate system-RAM tier across
PCIe. Results from the Mac must therefore be labelled **unified-memory**, not
"experts outside GPU memory."

The pinned OLMoE artifacts are appropriate sizes for this machine: Q6_K is
5.29 GiB and Q2_K is 2.39 GiB on disk. Q6_K leaves roughly 12.7 GiB of the
nominal 18 GB pool for macOS, KV cache, compute buffers, and applications.
That should support the 4,096-token single-request qualification comfortably,
but the eight-request contract can consume about 4 GiB of F16 KV plus compute
buffers and may encounter macOS memory pressure. Swap activity invalidates a
clean in-memory throughput comparison and must be recorded.

`spm-gguf-inspect --moe-summary` derives the Q6_K expert traffic directly from
the pinned tensor directory without reading model payloads. Its 48 expert
tensors contain 5,284,823,040 packed bytes. One expert across one layer is
5,160,960 bytes; top-8 routing across 16 layers therefore requests exactly
660,602,880 expert bytes per token before cache or batch reuse. At a
hypothetical sustained 5 GB/s, storage alone caps batch-one near 7.6 tokens/s;
decode and expert arithmetic can only lower that ceiling. Serving several
routed activations during the same expert read is the intended escape hatch.

Run the same hashes, prompts, context lengths, and request counts on the Mac in
three modes where supported:

1. llama.cpp Metal with the model fully GPU-offloaded;
2. llama.cpp CPU-only;
3. the Rust ordered expert implementation on CPU.

Record aggregate and per-request throughput, latency distributions, peak
process resident memory, `memory_pressure`, swap before/after, and wall energy.
On macOS, `powermetrics` can provide package-oriented telemetry when run with
the required privileges; it is not directly interchangeable with NVIDIA-only
board power. Report elapsed joules and measurement scope rather than comparing
unlabelled watt readings.

The Mac also supports a more ambitious capacity experiment: use a model whose
complete weights exceed 18 GB, keep the attention path, KV cache, activations,
router, and small shared tensors in unified memory, and keep the bulk MoE
expert store on the internal SSD. After routing, read only selected experts
into one or two bounded reusable buffers, execute them on CPU, discard or
reuse the buffers, and pass their outputs back to the shared activation path.
The complete expert store must never become resident at once.

```text
SSD: packed expert store (> available unified memory)
          |
          | selected expert extents, sequential reads
          v
bounded CPU buffer A/B -> decode + expert MACs on CPU
          |                         |
          +------ expert output ----+
                                    v
unified-memory activations -> Metal attention/non-expert path
                                    |
                                    v
                              KV cache in memory
```

This would validate the **bounded-residency mechanism** on Apple hardware, but
not a discrete CPU-RAM-versus-VRAM placement claim: Metal and the CPU still
share the same physical memory. It may nevertheless be an unusually favorable
hybrid because expert outputs and activations do not need an explicit PCIe
copy between separate CPU and GPU memories.

Ordinary file-backed `mmap` is not sufficient evidence of SSD streaming.
macOS may retain touched expert pages in the unified file cache until memory
pressure evicts them, making later tokens appear much faster while consuming
the capacity the experiment claims to save. The harness must report resident
and cached memory and use an explicit bounded-read policy. Where supported,
compare ordinary buffered reads with cache-advisory or no-cache file access;
label both precisely rather than claiming physical SSD traffic from requested
byte counts. Cold-cache trials need a documented, repeatable cache-control
procedure or must remain labelled first-touch proxies.

The oversized Mac matrix should therefore measure:

1. bounded buffer sizes of one expert, two experts, and a small prefetched
   window;
2. batch/request counts that reuse each selected expert read;
3. requested expert bytes versus observable physical-storage bytes;
4. SSD read throughput, CPU decode/MAC time, Metal time, and synchronization;
5. peak process/unified-memory pressure, compression, and swap;
6. latency, throughput, task correctness, and scoped wall energy;
7. warm-cache, bounded-cache-advisory, and defensible cold/first-touch cases.

The Mac therefore answers three useful questions: whether the GGUF/Rust
implementation is portable beyond x86/CUDA, whether avoiding discrete PCIe
transfers changes the break-even point, and whether an oversized MoE can run
with genuinely bounded expert residency backed by a fast consumer SSD.
large12 and the planned dual-socket `big72` test remain necessary to establish
the discrete-GPU capacity and CPU/DRAM placement comparisons.

### Recent oversized candidate search

The target is a current routed MoE with a credible quant slightly larger than
18 GB, not merely any model whose parameter count is large. Official configs
and current GGUF inventories were checked on 2026-09-04.

**Primary candidate: Gemma 4 26B-A4B-it.** The official configuration has 128
experts, top 8, 30 layers, and ordinary sliding/full attention rather than
Mamba. Bartowski's Q5_K_M GGUF is 19,319,198,848 bytes (18.0 GiB) and Q6_K is
22,862,577,280 bytes (21.3 GiB). Q5_K_M is the cleanest first Mac experiment:
it is only slightly above nominal capacity once macOS and runtime state are
included, and it avoids conflating expert streaming with a new state-space
implementation. Q6_K is the stronger follow-up after the mechanism works.
The Q5_K_M pin is repository revision
`10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d`, SHA-256
`6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493`.

**Deferred hybrid candidate: NVIDIA Nemotron 3.5 Lightning 30B-A3B.** It is a
routed MoE with 30B total and about 3B active parameters. The ggml-org Q4_0 is
18,898,091,584 bytes; Unsloth's low-bit artifacts start around 19.43 GB and
Q4-class artifacts are about 24.5--25.5 GB. This is nearly ideal in size, but
the official `NemotronH` configuration includes Mamba heads. It belongs in a
later hybrid-attention/state-space experiment, not the first expert-stream
validation.

**Not selected: Qwen3.8-27B.** It is current, but its official configuration
is dense and has no routed experts. Its Q4_K_M is 18,973,870,432 bytes, making
it an interesting bounded *layer-streaming* control, not an MoE expert-stream
target. Qwen3.8-Flash-Next is MoE (512 experts, top 10) but even its IQ1 GGUF is
roughly 72.5--74.5 GB and its architecture uses linear-attention/state-space
machinery. It is both much larger than requested and a future architecture.

Sources:

- <https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json>
- <https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/tree/10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d>
- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json>
- <https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/main>
- <https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json>
- <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/main>
- <https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json>

Plan for roughly twice the GGUF size while developing because both the source
GGUF and an expert-major streaming layout may coexist. Gemma Q5_K_M therefore
needs about 45--50 GB free for source, repack, logs, and safe temporary space;
Q6_K needs about 55--60 GB. A finalized importer may stream-repack into a
temporary destination, verify hashes and extents, and then allow the source to
be archived or removed; the benchmark itself must not require two permanent
copies.
