# Claim scorecard: what would success mean?

## One-sentence conclusion

**Today emufpga proves that Granite 3.1 1B-A400M Q6_K expert weights can be
processed serially with bounded parameter residency and preserved model
outputs; it does not yet prove that a small/old GPU can serve an oversized MoE
faster than conventional CPU/System-RAM offload.**

> **Validated oversized-model serving advantage: not measured.**

An “agent” must be an independent request stream, not several tokens from one
prompt. A small deterministic task smoke suite has now been measured for the
conventional oversized baseline. Energy at the wall has not.

## Evidence ledger

| Statement | Status | Evidence |
| --- | --- | --- |
| Granite Q6_K experts stream in expert-ID order | **Measured** | All 24 MoE layers; zero mid-region seeks |
| Streamed full forward preserves the oracle result | **Measured** | Same top-1, 9/10 top logits, absolute error below 0.002 |
| Five prompt prefixes show correlated route reuse | **Measured, small sample** | 23 tokens × 24 layers; B6 union 21/32 versus 26.3/32 independent estimate |
| Selected B1 layout reduces expert bytes | **Measured** | 42.21 MB to 10.65 MB per layer |
| Async buffering accelerates this host | **Mixed measured result** | Seven-run cases range from −5.5% to +11.7%; cache-bypass proxies +2–5% |
| Same-quant OLMoE placement saves VRAM | **Measured conventional baseline** | Q6_K saves 4,995 MiB peak VRAM; Q2_K saves 2,220 MiB across three-run sweeps |
| Real OLMoE Q6_K selected experts execute serially | **Measured mechanism** | Layer 0, batches 1/2/4/8; packed stream agrees with direct GGUF oracle within 0.00000004 |
| Gemma 4 Q5_K_M exceeds this GPU | **Measured capacity failure** | All-GPU allocation requested 18,409 MiB and failed on the 16,311 MiB RTX 5060 Ti |
| Oversized Gemma conventional offload serves independent requests | **Measured baseline smoke** | 20/30 layers on GPU; 128+16 tokens; 3 runs; aggregate 3.45/4.86/5.60/7.11 tok/s at 1/2/4/8 requests |
| Conventional CPU expert placement improves complete service time | **Measured negative** | With 3,840 prompt + 256 generated tokens, median end-to-end time is 2.7–5.9× Q6 all-GPU and 2.1–4.4× Q2 all-GPU |
| Lower-bit placement is automatically faster | **Measured negative** | Q2 CPU experts improve generation-only throughput at 1–4 requests, but 7–9× slower prefill reverses the end-to-end conclusion |
| Independent agents share a streamed pass | **Architecture-tested, not end-to-end measured** | Synthetic batching proves reuse math; Granite batches are prompt tokens, not agents |
| FPGA throughput, watts, and results/kWh | **Simulated/projected only** | Hardware-shaped cycles; no synthesized clock, physical link, or power trace |
| MCU/PIO improves tensor throughput | **Not claimed** | Proposed only for framing, backpressure, DMA control, and timestamps |

Granite remains the strict serial correctness vehicle. OLMoE Q6_K and Q2_K now
provide a same-artifact llama.cpp placement baseline, but they also fit the
GPU and the CPU placement is not the project's ordered bounded stream. Gemma 4
26B-A4B-it Q5_K_M (19,319,198,848 bytes) is the first artifact measured to
exceed the GPU capacity. Conventional 20-layer GPU offload now supplies the
comparison to beat: peak 13,592 MiB VRAM, 9,183 MiB process RSS, and 3.16 kJ of
GPU-board energy over model load plus the complete request sweep. At four
requests it misses the “good enough” target (1.40 rather than 2 generated
tok/s/request). Until Gemma runs through the serial path, the primary economic
value proposition remains unvalidated.

## The experiment that produces the requested headline

- Granite 3.1 1B-A400M Q6_K for serial correctness, OLMoE Q6_K/Q2_K for
  same-quant placement qualification, then pinned Gemma 4 26B-A4B-it Q5_K_M
  for the oversized capacity test.
- 1, 2, 4, and 8 independent requests; fixed 4K input and 256 greedy output
  tokens per request.
- The same task corpus and decoding policy for resident GPU, llama.cpp CPU/RAM
  offload, serial CPU workers, GPU expert staging, and eventually FPGA.
- Synchronized whole-system wall-power integration, including storage and idle,
  as a supporting reuse/economics measure rather than the primary objective.
- Report correct tasks/hour, tokens/s, joules/token, correct tasks/kWh, p50/p95
  latency, peak RAM/VRAM, and bytes moved.

`correct tasks/kWh = correct completed tasks / (integrated joules / 3,600,000)`.
Tokens/kWh is useful but cannot substitute for task correctness.

The desired eventual headline has this form: “On reused host H with small GPU G,
N independent agents ran oversized MoE M at quant Q with X correct tasks/hour
and Y aggregate tokens/s, versus Z on llama.cpp CPU/RAM offload, while using A
GiB VRAM and B watts.” No value in that sentence should be projected.

The current 128-input/16-output measurement is a harness and capacity smoke
test, deliberately shorter than the predeclared 4K+256 final contract. It must
not be substituted for that longer qualification.

## Predeclared verdicts

**Success:** correctness is statistically indistinguishable from the same
quantized reference; the model does not fit the small GPU's usable VRAM; the
hybrid serial path beats llama.cpp CPU/System-RAM offload on the same reused
host by at least 25% in correct completed tasks/hour; and an initial “good
enough” service objective of four concurrent agents, at least 2 generated
tokens/s each, is met. The result must survive three runs. Power, acquisition
cost, peak RAM/VRAM, and storage wear are reported but do not veto a capacity
and throughput win unless the operating cost is plainly unreasonable.

**Failure:** correctness regresses, the hybrid cannot run an otherwise
oversized model, it fails the service objective, or its confidence interval
does not support any throughput gain over ordinary CPU/RAM offload.

**Mixed:** it unlocks capacity but does not beat CPU offload, wins only above a
concurrency threshold, costs excessive energy, or wins on some reused hardware
generations but not others.
Mixed is plausible: old-hardware reuse is a multi-objective choice.

These thresholds are project policy, not facts. They may be revised before the
energy campaign, but not after seeing its result without recording the change.
