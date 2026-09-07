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

### Cold-cache I/O (scoped: 1 repetition, c1 and c8)

`bench-gemma4-cold-io` evicts the model's page-cache pages per-file with
`posix_fadvise(POSIX_FADV_DONTNEED)` (no root needed; not a global
`drop_caches`), then loads a fresh server for every cache/policy
combination. The model file lives on a rotational HDD (Seagate
ST4000NM0063, ext4). Eviction was verified effective: `resident_pages`
returned to 0 before each cold trial.

| Trial | Passed | Startup HDD read | Total phys read | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| c1 cold, resident | 1/1 | 3.21 GB | 8.43 GB | 11,707 MiB |
| c1 cold, reclaimed | 1/1 | 3.21 GB | 8.43 GB | 10,224 MiB |
| c1 warm, resident | 1/1 | 0 | 0 | 11,819 MiB |
| c1 warm, reclaimed | 1/1 | 0 | 0 | 8,951 MiB |
| c8 cold, resident | 8/8 | 3.21 GB | 11.13 GB | 14,403 MiB |
| c8 cold, reclaimed | 8/8 | 3.21 GB | 11.11 GB | 12,616 MiB |
| c8 warm, resident | 8/8 | 0 | 0 | 14,464 MiB |
| c8 warm, reclaimed | 8/8 | 0 | 0 | 12,203 MiB |

Findings, all consistent with the 5060 lane's cold-HDD qualification:

- **All trials pass** cold or warm, resident or reclaimed. Peak VRAM
  stayed 3372-3378 MiB throughout.
- **Cold demand grows sub-linearly with concurrency**: 8.43 GB at one
  request, 11.13 GB at eight -- so bytes per passing task fall sharply as
  the eight-request route union reuses experts. The 5060 lane measured
  8.12 GB (c1) and 10.70 GB (c8); this box's 8.43 / 11.13 GB land in the
  same place. Lazy loading keeps startup reads to 3.21 GB, not the full
  19.3 GB.
- **Reclamation lowers peak RSS but not physical bytes**: it saved about
  1.48 GiB at c1 (11,707 -> 10,224) and 1.79 GiB at c8 (14,403 ->
  12,616), while cold read totals were unchanged (8.43 = 8.43; 11.13 vs
  11.11). Same conclusion the 5060 lane reached: reclamation is a
  memory/latency tradeoff, not a disk-traffic reduction.
- These reads are demand-paged mmap faults, not a purpose-built
  sequential expert stream -- the same negative-control caveat the 5060
  lane recorded.

Derived data: [gemma4-q5km-coldio-rtx3060-scoped.json](data/gemma4-q5km-coldio-rtx3060-scoped.json).

### Cold-cache I/O (widened: 3 repetitions, c1/2/4/8)

The full campaign matching the 5060 lane's cold-HDD qualification: 3
repetitions x concurrency 1/2/4/8 x cold/warm x resident/reclaimed = 48
trials, 180 executable tasks. Means across the three repetitions:

| Concurrency | Cold read (mean) | Bytes per passing task | Peak RSS resident -> reclaimed | Cold completion rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.43 GB | 8.43 GB | 11,707 -> 10,219 MiB | 30.3 tasks/h |
| 2 | 9.54 GB | 4.77 GB | 12,794 -> 11,182 MiB | 50.2 tasks/h |
| 4 | 10.52 GB | 2.63 GB | 13,777 -> 11,248 MiB | 89.0 tasks/h |
| 8 | 11.13 GB | 1.39 GB | 14,410 -> 12,577 MiB | 139.3 tasks/h |

- **All 180 executable evaluations passed**, cold or warm, resident or
  reclaimed -- the same headline the 5060 lane reported.
- **Cold demand grows sub-linearly** (8.43 -> 11.13 GB from c1 to c8),
  so bytes per passing task fall 6x (8.43 -> 1.39 GB) as the eight-request
  route union reuses experts. The 5060 lane measured 8.12 -> 10.70 GB and
  1.34 GB per task at c8; this box lands in the same place.
- **Reclamation saved 1.49 to 2.53 GiB peak RSS** across concurrencies
  without reducing cold disk bytes (the resident and reclaimed cold reads
  match at each concurrency) -- the 5060 lane saw 1.47 to 2.79 GiB. It is
  a memory/latency tradeoff, not a disk-traffic reduction.
- **Completion rate rises with concurrency** (30.3 -> 139.3 cold resident
  tasks/hour), the expected amortization; the 5060 lane rose 23.2 ->
  109.8, and this dual-Xeon box runs somewhat faster in absolute terms.
- Reads remain demand-paged mmap faults, not a purpose-built sequential
  expert stream -- the same negative control the 5060 lane flagged.

Derived data: [gemma4-q5km-coldio-rtx3060-wide.json](data/gemma4-q5km-coldio-rtx3060-wide.json).

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
