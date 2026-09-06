# Re-running the too-large-to-fit workaround on other machines

The Gemma-4 serial-expert experiments (docs/gemma4-serial-experts.md)
were first measured on `large12`, an RTX 5060 16 GiB (Blackwell,
compute capability 12.0, `sm_120`) box whose scratch disk was mounted
at `/disk1`. This note records what it takes to reproduce them on a
different machine -- the immediate target being an RTX 3060 12 GiB
(Ampere, compute capability 8.6, `sm_86`) box that cloned the repo onto
a large data disk instead of `/disk1`.

The experiment is a capacity claim, not a speed claim: a 19.3 GB
Q5_K_M model that does not fit in VRAM is made to run by keeping
attention/KV/shared work on the GPU (about 4.17 GiB VRAM measured on
large12) while the MoE experts stay in CPU/host memory with lazy
`MADV_DONTNEED` reclamation. Because the GPU side is only ~4.2 GiB, a
12 GiB card has ample room; the 16-to-12 GiB reduction does not touch
the workaround's ceiling.

## What was machine-specific, and what changed

The scripts assumed two things that are not portable. Both are now
parameterised (branch `feat-3060`), backward compatible on large12.

| Assumption | Was | Now |
| --- | --- | --- |
| GPU architecture | `-DCMAKE_CUDA_ARCHITECTURES=120a` hardcoded | `scripts/cuda-arch.sh` autodetects from `nvidia-smi` (override `CUDA_ARCH`); 120 maps to `120a`, 86 to `86` |
| Scratch disk | `/disk1/tmp/...` | `${SPM_SCRATCH:-/disk1/tmp}` |
| Source checkout | `/disk1/github/...` | `${SPM_SRC:-/disk1/github}` |
| Model file identity | pinned `device:inode:size` | pinned size only (device and inode are host-specific); SHA-256 still verified by callers |
| CUDA build dir | `llama.cpp-sm120-build` | `llama.cpp-sm${arch}-build` (so large12 still resolves to `sm120`) |

CPU core count and RAM size were already portable: builds use
`-j "$(nproc)"` and llama.cpp defaults its thread count to the host.
This box has 503 GiB RAM, which dwarfs the ~14-19 GiB expert residency;
core count only affects throughput, not whether the workaround runs.

## Test plan on the RTX 3060 box

Set the scratch and source roots once for the session (any disk with
~40 GiB free for the model plus a CUDA build):

```sh
export SPM_SCRATCH=/mnt/storage1/spm-scratch
export SPM_SRC=/mnt/storage1/github
mkdir -p "$SPM_SCRATCH" "$SPM_SRC/ggml-org"
```

1. **Fetch the identical model.** Same pinned artifact as large12:
   `google_gemma-4-26B-A4B-it-Q5_K_M.gguf`, SHA-256
   `6b4f8074239f72543997a950ff6ae4509553c599f5b432748309ff3438416493`,
   19,319,198,848 bytes, from
   `bartowski/google_gemma-4-26B-A4B-it-GGUF` revision
   `10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d`.

   ```sh
   huggingface-cli download bartowski/google_gemma-4-26B-A4B-it-GGUF \
     google_gemma-4-26B-A4B-it-Q5_K_M.gguf --revision 10f3b41bcf8d3047f4e136e7197ffc2dd1654c9d \
     --local-dir "$SPM_SCRATCH/gemma-4-26b-a4b"
   sha256sum "$SPM_SCRATCH/gemma-4-26b-a4b/google_gemma-4-26B-A4B-it-Q5_K_M.gguf"
   ```

2. **CPU layer smoke first (cheap, no GPU, no llama.cpp).** Validates
   the expert-layer arithmetic against the direct GGUF path. On large12
   the max error was 0.00000381. This only needs the model.

   ```sh
   scripts/verify-gemma4-expert-smoke 1
   scripts/verify-gemma4-expert-smoke 8
   ```

3. **Clone and build patched llama.cpp for this GPU.** The build script
   now compiles for the detected arch (`sm_86` here).

   ```sh
   git clone https://github.com/ggml-org/llama.cpp "$SPM_SRC/ggml-org/llama.cpp"
   git -C "$SPM_SRC/ggml-org/llama.cpp" checkout 4d9176092d00586775af140581bb0b558ddc4389
   scripts/build-llama-gemma4-lazy-experts
   ```

4. **End-to-end workaround (the capacity run).** Experts on CPU, GPU
   attention/KV, lazy reclamation.

   ```sh
   LLAMA_BUILD="$SPM_SCRATCH/llama.cpp-gemma4-lazy-sm86-build" \
     scripts/bench-gemma4-offload
   scripts/bench-gemma4-cold-io          # cold/warm x resident/reclaimed campaign
   ```

5. **Compare against large12** in docs/gemma4-serial-experts.md.

## What should reproduce, and what should not

- **Reproduce:** the capacity success -- a model that fails resident
  GPU allocation generates correct answers; peak VRAM well under 12 GiB;
  all executable Rust tasks pass; reclamation lowers peak RSS.
- **Will differ:** throughput (tok/s, TTFT) and energy -- the 3060 is a
  slower, different-generation part with a different core count. These
  are usability numbers, not the capacity criterion.
- **Do not expect bit-identical text vs large12.** CUDA numerics can
  differ across GPU architectures, and docs/gemma4-serial-experts.md
  already records that continuations vary between placements at
  temperature zero. The within-run reclaim-vs-resident logit A/B
  (bit-identical on large12) is a same-binary comparison and should
  still hold here; the cross-machine comparison should be judged on
  task pass rate, not exact tokens. The `sm120`-tagged parity golden
  `tools/llama-logits/qwen3-8b-q6-capital-sm120.golden` is not a valid
  oracle on Ampere -- regenerate an `sm86` golden if running the qwen3
  parity path.
