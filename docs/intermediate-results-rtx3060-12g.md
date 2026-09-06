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

As of 2026-09-06 the environment and build are ready and the pinned
model is still downloading. No inference results are claimed yet; the
results table below carries PENDING until each experiment runs.

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

### Remaining (GPU end-to-end)

| Experiment | Metric | RTX 3060 12G |
| --- | --- | --- |
| End-to-end offload (`bench-gemma4-offload`) | peak VRAM | PENDING |
| End-to-end offload | executable tasks passed | PENDING |
| End-to-end offload | aggregate tok/s, c1..c8 | PENDING |
| Reclaimed vs resident logits (same binary) | bit-identical? | PENDING |

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
