# Gemma-4 oversized MoE offload baseline

## Pinned artifact and capacity result

- Model: Gemma-4 26B-A4B Instruct, 30 layers, 128 experts, top 8.
- Quant: Bartowski Q5_K_M, revision
  `10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d`.
- File: 19,319,198,848 bytes; SHA-256
  `6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493`.
- llama.cpp: `4d9176092d00586775af140581bb0b558ddc4389`, CUDA
  SM120a build.

`spm-gguf-inspect --moe-summary` reports 17,287,086,080 expert tensor bytes,
4,501,845 bytes per expert/layer on average, and 135,055,360 bytes for one
expert across all layers. An attempted `-ngl 999` load requested an
18,409.21 MiB CUDA buffer and failed allocation on the RTX 5060 Ti's 16,311
MiB. This is the project's first measured genuinely oversized model.

## Conventional control

`scripts/bench-gemma4-offload` keeps 20 of 30 repeating layers on GPU and runs
three repetitions at 1, 2, 4, and 8 independent requests. Each request has
exactly 128 input and 16 forced output tokens. Eight distinct elementary tasks
provide a correctness smoke check. This shorter contract validates the harness
and capacity baseline; the final qualification remains 4K+256 where hardware
capacity permits.

| Requests | Correct | Aggregate generated tok/s | Per-request tok/s | TTFT p50 | TTFT p95 | Inter-token p50 | Inter-token p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3/3 | 3.45 | 3.45 | 3.61 s | 3.91 s | 66 ms | 149 ms |
| 2 | 6/6 | 4.86 | 2.43 | 3.83 s | 4.63 s | 163 ms | 261 ms |
| 4 | 12/12 | 5.60 | 1.40 | 6.19 s | 9.87 s | 296 ms | 359 ms |
| 8 | 21/24 | 7.11 | 0.89 | 11.50 s | 11.88 s | 440 ms | 524 ms |

Peak process RSS was 9,182.9 MiB and peak VRAM was 13,592 MiB. Integrated GPU
board energy was 3,162 J over 123.2 seconds spanning server load and the whole
sweep. That energy excludes CPU, DRAM, disks, motherboard, fans, and PSU loss,
so it cannot support a tasks/kWh or whole-machine efficiency claim.

The result is a capacity success for ordinary offload and a service warning:
aggregate throughput rises with concurrency, but per-request throughput and
latency degrade. At four requests, 1.40 tok/s/request misses the project's
initial 2 tok/s/request “good enough” threshold. The serial implementation
must beat this same-quant control; it has not done so yet.

Derived data: [`data/gemma4-q5km-offload.json`](data/gemma4-q5km-offload.json).
Raw responses, logs, and 200 ms telemetry are intentionally outside Git.
