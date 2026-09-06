# Intermediate results: RTX 3060 12G reproduction

This is the RTX-3060-12G lane's copy of the oversized-MoE workaround
results. It is deliberately a standalone, hardware-labeled file so it
merges cleanly beside the original RTX-5060-16G measurements in
[intermediate-results.md](intermediate-results.md) and
[gemma4-serial-experts.md](gemma4-serial-experts.md) rather than
conflicting with them. The shared methodology lives in those documents;
this file records only what was measured on this box.

## Concise status

Reproducing the too-large-to-fit Gemma-4 Q5_K_M workaround on a second,
unrelated machine to test whether the capacity *result* travels, not
just the ported code. The workaround keeps attention, KV, and shared
weights on the GPU and streams the MoE experts from host memory with
lazy reclamation, so the GPU-resident footprint (about 4.17 GiB on the
5060 lane) is far below 12 GiB and the card size is not the ceiling.

As of 2026-09-06 the reproduction is complete for the two primary
checks: the CPU layer-0 expert smoke matches the 5060 lane bit-for-bit,
and the end-to-end GPU offload run is a capacity success (all requests
correct, peak VRAM 3384 MiB on the 12 GiB card). Optional follow-ups
(reclamation logit A/B, 256-token executable sweep, cold-cache campaign)
are listed but not yet run.

## Environment (this box)

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB (11907 MiB reported), Ampere `sm_86` |
| CUDA toolkit | 13.0 (nvcc V13.0.88) |
| CPU | 2x Intel Xeon E5-2697 v4, 36 physical cores / 72 threads |
| RAM | 503 GiB |
| Scratch disk | /mnt/storage1 (via SPM_SCRATCH), 2.4 TiB free |

For contrast, the original lane ran on `large12`: RTX 5060 16 GiB
(Blackwell `sm_120`), scratch on `/disk1`. Its numbers stay in the
5060 documents; this file does not restate them so the two lanes do not
have to be kept in sync by hand.

## Pinned inputs (identical bytes to the study)

| Input | Pin |
| --- | --- |
| Model | `google_gemma-4-26B-A4B-it-Q5_K_M.gguf`, SHA-256 `6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493`, 19,319,198,848 bytes, from `bartowski/google_gemma-4-26B-A4B-it-GGUF` rev `10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d` |
| llama.cpp | revision `4d9176092d00586775af140581bb0b558ddc4389`, patched by `patches/llama.cpp-gemma4-lazy-experts.patch`, built for `sm_86` |

The port that made this reproducible on non-`/disk1`, non-Blackwell
hardware is described in [rtx3060-port.md](rtx3060-port.md).

## Provisioning status

| Item | Status |
| --- | --- |
| Rust tree builds (`spm-granite-moe`) | DONE |
| llama.cpp cloned + checked out at pinned rev | DONE |
| llama.cpp CUDA build for `sm_86` | DONE (llama-server, llama-cli link; device enumerates as `CUDA0: RTX 3060`) |
| Pinned model download + SHA-256 verify | DONE (sha256sum -c OK) |

## Measured results

Nothing here is claimed until it is measured on this box.

### Layer-0 selected-expert smoke (CPU, `verify-gemma4-expert-smoke`)

This path is deterministic CPU code (the `spm-granite-moe` Rust engine
against exact GGUF bytes), so an exact match to the 5060 lane is the
correct expectation, and that is what was observed. Routing, distinct
fetches, stream bytes, the 176-byte resident footprint (one Q5_K block),
and the maximum error are bit-for-bit identical to the 5060 lane's
layer-0 table in [gemma4-serial-experts.md](gemma4-serial-experts.md).

| Batch | Assignments | Distinct fetches | Stream MB | Resident B | Max abs error (3060) | Max abs error (5060) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 41.14 | 176 | 0.00000191 | 0.00000191 |
| 2 | 16 | 12 | 61.70 | 176 | 0.00000381 | 0.00000381 |
| 4 | 32 | 20 | 102.84 | 176 | 0.00000381 | 0.00000381 |
| 8 | 64 | 33 | 169.69 | 176 | 0.00000381 | 0.00000381 |

CPU timings differ from the 5060 lane, as expected on a different host
(dual Xeon E5-2697 v4): direct-path 144/228/396/699 ms and scalar serial
stream 455/738/1305/2313 ms at batch 1/2/4/8. These are usability
numbers, not correctness, and the scalar stream loop is unoptimized.

### End-to-end offload capacity run (GPU, `bench-gemma4-offload`)

Configuration: patched llama.cpp built for `sm_86`, `PLACEMENT=experts-cpu`
(MoE experts on CPU), `-ngl 999` (all non-expert layers on GPU),
`--lazy-mode on`, resident policy (no `MADV_DONTNEED`), warmup off,
`RUNS=1`, output 16 tokens, the `gemma4-correctness-tasks` corpus.

The 19.3 GB Q5_K_M model cannot be held resident on a 12 GiB card
(all-GPU allocation is arithmetically impossible, and the 5060 lane
recorded it failing even at 16 GiB). The workaround nonetheless
generated correct answers:

| Concurrency | Correct | Peak VRAM | Aggregate tok/s | Per-request tok/s | TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | -- | 5.05 | 5.05 | 2507 ms |
| 2 | 2/2 | -- | 12.14 | 6.08 | 1675 ms |
| 4 | 4/4 | -- | 18.61 | 4.66 | 2065 ms |
| 8 | 8/8 | 3384 MiB | 19.28 | 2.41 | 4466 ms |

- **Capacity success reproduced.** All 15 requests correct; peak GPU
  memory was 3384 MiB (sustained 3346-3384 across the run; 43 MiB idle),
  well under the 12 GiB card and in the same range as the 5060 lane's
  ~4170 MiB offload peak. Peak process RSS was 15,178 MiB (~14.8 GiB),
  overwhelmingly file-backed expert mappings, which the 503 GiB host RAM
  absorbs easily.
- **Throughput is not comparable to the 5060 bounded table.** That table
  used 256 output tokens and the executable corpus; this run used 16
  tokens and the correctness corpus, so the tok/s here characterize this
  configuration only. Aggregate throughput still rises with concurrency
  (5.05 -> 19.28 tok/s from c1 to c8), the expected expert-reuse
  amortization.
- The `sm_86` build ran with zero CUDA errors or kernel-launch failures.

### Reclaimed vs resident logit equivalence (same binary)

The logit probe (`tools/llama-logits`) evaluated one fixed prompt twice
through the same `sm_86` patched binary and native Q5_K_M kernels, lazy
mode on, `-ngl 999`: the control left selected expert mmap pages
resident; the candidate called `MADV_DONTNEED` after each expert
(`GGML_MUL_MAT_ID_DONTNEED=1`).

| Property | Result |
| --- | --- |
| Raw final-logit array | 262,144 floats = 1,048,576 bytes (both) |
| SHA-256 (resident) | `2c79d245e43865fc9ece2ea00fc0aadf4c66d1cdce45e193bbeaaa8ef4f786f6` |
| SHA-256 (reclaimed) | identical |
| Bit-for-bit identical | YES (`cmp` clean; top-10 tokens/logits match to the hex bit pattern) |
| Peak VRAM during probe | 2188 MiB |

This reproduces the 5060 lane's finding -- reclamation does not perturb a
single final-logit bit -- now on Ampere `sm_86`. The absolute SHA differs
from the 5060 lane only because the prompt differs; the invariant under
test is resident-vs-reclaimed equality within one binary, which holds.
Derived data: [gemma4-q5km-logit-equivalence-rtx3060.json](data/gemma4-q5km-logit-equivalence-rtx3060.json).

### Executable-code sweep (256-token, Docker-sandboxed)

The 256-token executable corpus (`gemma4-rust-executable-v1`) through the
same experts-CPU / GPU-attention / lazy path, with each response compiled
and run in a locked-down `rust:1.89-slim-bookworm` container (no network,
read-only root, memory/CPU/PID limits, timeouts). Resident policy, one
repetition.

| Concurrency | Compiled | Passed | Strict format | Tokens | Wall ms (mean/req) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 1/1 | 1/1 | 62 | 4958 |
| 2 | 2/2 | 2/2 | 1/2 | 127 | 5613 |
| 4 | 4/4 | 4/4 | 1/4 | 289 | 7591 |
| 8 | 8/8 | 8/8 | 1/8 | 541 | 15299 |

- **All 15 generated programs compiled and passed** in the sandbox, the
  same executable outcome the 5060 lane reported (which ran more
  repetitions and both policies). Peak VRAM 3380 MiB, peak RSS ~14.3 GiB
  -- consistent with the correctness-corpus run.
- **Strict function-only formatting is unstable** (1 strict response per
  batch) even though every executable result was sound -- the 5060 lane
  saw the same effect. Text compliance is not a reliable signal;
  compile-and-run is.
- This is one warm resident repetition on 15 tasks, not the 5060 lane's
  multi-repetition resident-and-reclaimed matrix; it confirms the
  executable outcome travels, not a full statistical re-run.

Derived data: [gemma4-q5km-executable-rtx3060.json](data/gemma4-q5km-executable-rtx3060.json).

Still not run on this box: a cold-cache campaign (`bench-gemma4-cold-io`,
which needs `drop_caches` / sudo).

## Expectations (to be confirmed or refuted, not assumed)

- **Should reproduce:** the capacity success -- a model that fails
  resident GPU allocation still generating correct answers, with peak
  VRAM well under 12 GiB, and all executable tasks passing.
- **Will differ:** throughput and energy -- the 3060 is a slower,
  different-generation part with a different core count than the 5060.
- **Not expected to match:** exact token-level output vs the 5060 lane,
  because CUDA numerics differ across GPU architectures. The within-run
  reclaimed-vs-resident logit A/B is a same-binary comparison and should
  still hold.
