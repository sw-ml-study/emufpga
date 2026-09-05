# Claim scorecard: what would success mean?

## One-sentence conclusion

**Today emufpga proves that real Granite, OLMoE, and oversized Gemma-4 expert
weights can be processed in selected serial order with bounded parameter
residency and close agreement to direct same-quant paths. Oversized Gemma now
completes a short end-to-end correctness smoke on the small GPU; coding-task
reliability is not yet established.**

> **Preliminary oversized-model capacity success: measured. Coding reliability: not measured.**

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
| Real Gemma Q5_K_M selected experts execute serially | **Measured layer mechanism** | Layer 0, batches 1/2/4/8; B8 collapses 64 assignments to 33 streams; max direct/stream error 0.00000381 |
| GPU state plus CPU experts completes Gemma generation | **Measured partition bridge** | 1/2/4/8 short tasks correct; 4,170 MiB VRAM but 19,514.5 MiB RSS; conventional mapped tensors, not bounded serial supply |
| Reclaimable selected-expert pages complete Gemma generation | **Measured preliminary capacity success** | Three 128+16 runs; 45/45 short tasks correct; 4,172 MiB VRAM; 15,070 MiB peak and 9,723 MiB mean RSS |
| Oversized Gemma conventional offload serves independent requests | **Measured baseline smoke** | 20/30 layers on GPU; 128+16 tokens; 3 runs; aggregate 3.45/4.86/5.60/7.11 tok/s at 1/2/4/8 requests |
| Conventional CPU expert placement improves complete service time | **Measured negative** | With 3,840 prompt + 256 generated tokens, median end-to-end time is 2.7–5.9× Q6 all-GPU and 2.1–4.4× Q2 all-GPU |
| Lower-bit placement is automatically faster | **Measured negative** | Q2 CPU experts improve generation-only throughput at 1–4 requests, but 7–9× slower prefill reverses the end-to-end conclusion |
| Independent agents share a streamed pass | **Layer-measured, not end-to-end** | Real Gemma routing at B8 gives 1.94 assignments per distinct expert fetch; batches are concurrent activations, not coding sessions |
| FPGA throughput, watts, and results/kWh | **Simulated/projected only** | Hardware-shaped cycles; no synthesized clock, physical link, or power trace |
| MCU/PIO improves tensor throughput | **Not claimed** | Proposed only for framing, backpressure, DMA control, and timestamps |

Granite remains the strict serial correctness vehicle. OLMoE Q6_K and Q2_K now
provide a same-artifact llama.cpp placement baseline, but they also fit the
GPU and the CPU placement is not the project's ordered bounded stream. Gemma 4
26B-A4B-it Q5_K_M (19,319,198,848 bytes) is the first artifact measured to
exceed the GPU capacity. Conventional 20-layer GPU offload now supplies the
practical control: peak 13,592 MiB VRAM, 9,183 MiB process RSS, and 3.16 kJ of
GPU-board energy over model load plus the complete request sweep. At four
requests it provides 1.40 generated tok/s/request. Gemma's layer-0 experts run
through the standalone serial path, and complete inference succeeds through a
Linux mmap reclamation prototype.
This validates the short-contract capacity proposition, not coding reliability
or the longer qualification.

## The experiment that validates the capacity claim

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
oversized MoE M at quant Q—which fails resident allocation—completed the
same-quant correctness suite with N concurrent agents while using A GiB VRAM.”
Throughput, latency, watts, and CPU/RAM offload belong beside that result so
users understand practicality; they are not the success gate.

The current 128-input/16-output measurement is a harness and capacity smoke
test, deliberately shorter than the predeclared 4K+256 final contract. It must
not be substituted for that longer qualification.

## Predeclared verdicts

**Success:** the model demonstrably does not fit the small GPU's usable VRAM;
the hybrid serial path completes end-to-end generation; greedy outputs/logits
agree with the same-quant reference within the declared gate; the coding task
suite shows no reliability regression; and peak GPU residency stays within the
budget. The result must survive three runs. Throughput, concurrency, latency,
power, acquisition cost, RAM, and storage wear are reported as usability and
economics—not used to redefine capacity success as failure.

**Failure:** correctness/reliability regresses, the hybrid cannot complete
generation, or it exceeds the declared GPU memory budget.

**Mixed:** it unlocks capacity correctly but has severe throughput, latency,
energy, RAM, storage-wear, or concurrency limits; or it works only on some
reused hardware generations.
Mixed is plausible: old-hardware reuse is a multi-objective choice.

These thresholds are project policy, not facts. They may be revised before the
energy campaign, but not after seeing its result without recording the change.
