# Analysis: what HLS4ML changes—and does not change—for emufpga

Paper reviewed: [hls4ml: A Flexible, Open-Source Platform for Deep Learning
Acceleration on Reconfigurable Hardware](https://arxiv.org/html/2512.01463v1)
(arXiv:2512.01463v1, 1 December 2025).

## Bottom line

The paper supports emufpga's **method and architectural vocabulary**, not its
performance proposition. hls4ml provides real precedents for reusing a small
number of arithmetic units, FIFO-connected dataflow, bit-exact precision
propagation, and post-place-and-route reporting. It does not validate fetching
selected billion-parameter MoE experts from a sequential external store.

The most important lesson is negative: more serialization reliably raises the
initiation interval, but does not reliably reduce fabric resources. Compiler,
precision, target device, and physical routing determine the result. Therefore
emufpga's simulated cycles, hypothetical clock, and bandwidth ceilings must
remain projections until a concrete design has been placed, routed, and
measured.

## Transfer matrix

| hls4ml finding | Application here | Confidence / boundary |
| --- | --- | --- |
| A reuse factor (RF) trades parallel multipliers for additional cycles; for an M×N matrix hls4ml limits multipliers per cycle to M×N/RF. | Treat decoder/MAC lanes and cycles-per-group as an explicit reuse-factor-like axis. Graph capacity, initiation interval, and bandwidth together. | **Directly applicable as a model.** It is not a device resource estimate. |
| The Resource strategy partitions on-chip weights and feeds a controlled number of MAC units per cycle. | Our Q6_K chunk decoder and MAC lanes are a related resource-reuse pipeline. Keep the schedule and packed representation separate from the execution backend. | **Structurally useful, physically different.** emufpga streams most weights from off-chip storage rather than retaining the matrix in BRAM. |
| `io_stream` joins layers with FIFOs; occupancy observed in RTL co-simulation can right-size FIFO depth. | Track FIFO high-water marks and producer/consumer stalls in simulation. Later, size buffers from traces rather than a guessed constant. | **Directly useful methodology.** hls4ml's FIFO is between operators, not proof that external expert storage will sustain the required rate. |
| Precision is propagated through the graph and accumulator widths are conservatively inferred to avoid overflow. | Add explicit activation, scale, product, and accumulator formats to the Rust IR; test bit-exactness and overflow independently of weight decoding. | **Directly useful.** GGUF Q6_K defines packed weights, not the whole arithmetic contract. |
| Hardware-aware quantization/pruning can outperform uniform compression. | Explore per-layer/per-expert precision only after preserving an accuracy oracle. Structured expert/block layouts matter more than arbitrary sparsity for a sequential reader. | **Promising experiment, not a current result.** It requires retraining or calibrated requantization and quality measurements. |
| IR, optimizer passes, and backend templates isolate model meaning from implementation choices. | Evolve `.spm` toward an inspectable schedule/precision IR while keeping Rust parsing and emulation independent of any future HLS/RTL backend. | **Strong software-design precedent.** Adopting hls4ml itself is unnecessary. |
| Post-place-and-route results vary sharply across devices and compiler choices; some ostensibly valid designs fail placement. | Preserve the current labels: measured CPU, derived math, projected bandwidth, and abstract cycles. Never turn a clock input into an “FPGA result.” | **Directly applicable validation discipline.** |
| Surrogate resource models become useful only after training on many synthesis results. | A future estimator should be calibrated from our own target/device/tool results and report error bars. | **Later only.** There is no valid training corpus in this repository yet. |

## What cannot be imported as evidence

1. **Scale and workload.** The reported applications are predominantly compact
   classifiers with weights embedded in logic or on-chip memory. They do not
   exercise autoregressive decoding, a KV cache, dynamic top-k MoE routing, GGUF
   Q6_K block decoding, or multi-gigabyte expert streams.
2. **Published speed numbers.** Frequencies, nanosecond latencies, and resource
   percentages are tied to particular AMD/Intel parts, compiler versions,
   precisions, and tiny graphs. None calibrates emufpga's hypothetical clock.
3. **The memory value proposition.** `io_stream` describes transport between
   hardware operators. It does not establish the storage bandwidth, energy,
   first-token latency, or endurance economics of reading selected experts from
   NVMe, flash, host RAM, or HBM.
4. **MoE scheduling.** The paper supplies no measured selected-expert
   serialization, routing overlap, expert-cache policy, or batch union result.
   emufpga still needs those empirical distributions.
5. **Toolchain safety.** hls4ml's Python frontends do not make Python pickle
   checkpoints safe. This project should continue to ingest bounded,
   validated GGUF/`.spm` data in Rust; ONNX/QONNX can be an interchange option
   only with strict parsing and resource limits.
6. **Backend availability.** Feature support differs by backend, and the paper
   itself notes tool/version churn. This is especially relevant to non-AMD
   targets and is another reason not to make a vendor HLS flow foundational.

## Changes to the emufpga validation plan

The paper strengthens this sequence without requiring FPGA programming now:

1. Measure real Granite routing distributions across prompts, tokens, and
   batches. Replace uniform-independent routing projections with traces and
   percentiles.
2. Extend the simulator report with FIFO occupancy, producer/consumer stalls,
   decoder utilization, and an explicit reuse/lanes sweep.
3. Specify widths for weights, scales, activations, products, and accumulators;
   add overflow and bit-exact tests.
4. Benchmark storage tiers separately and distinguish cold, warm, sequential,
   and concurrent reads. The bandwidth ceiling is useful only when its input is
   measured for the intended medium.
5. Much later, compare one small Q6_K expert kernel—not a whole LLM—with an HLS
   or RTL implementation. Report synthesis and post-place-and-route separately,
   then measure the board including I/O and power.

Until steps 1–4 exist, the defensible proposition remains **capacity through
sequential access, conditionally**, not lower latency or superior throughput.
