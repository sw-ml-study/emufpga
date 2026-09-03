# Qwen3 Rust parity rung

This rung keeps the GGUF trust boundary in Rust, establishes an independent
numerical oracle, and implements the first hand-auditable model slice.

## Exact decoders

`spm-gguf` now decodes the two types present in the Qwen3 8B Q6_K artifact:
little-endian F32 and GGML Q6_K. Q6_K uses the upstream 256-element/210-byte
block layout (128 low-bit bytes, 64 high-bit bytes, 16 signed scales, and one
little-endian F16 block scale). Hand-checkable tests exercise all four packed
lanes, the scale, and malformed lengths. `read_tensor_bytes` only accepts a
validated `TensorInfo` and an explicit allocation cap.

This is data decoding, not Python pickle loading: it does not import code,
resolve globals, or execute constructors. Rust removes pickle's arbitrary-code
execution mechanism here, while bounds checks and checked arithmetic remain
necessary for malicious binary inputs. Bounded subrange reads allow a single
embedding row to be fetched without allocating the 487 MiB embedding tensor.

## One-block Rust slice

`spm-qwen3-slice` evaluates token embedding plus transformer block zero for a
single token. It implements RMS normalization, Q/K/V projections, per-head
Q/K normalization, grouped-query attention, output projection, residuals, and
the SiLU-gated MLP. At position zero, RoPE is the identity and single-element
causal softmax is exactly one, which deliberately makes this first slice easy
to audit.

```sh
cd components/cli
cargo run --release -p spm-qwen3-slice -- \
  /disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf 785
```

For token 785 (the first oracle prompt token), the current deterministic stats
are:

```text
embedding len=4096 min=-0.35156250 max=0.34388733 rms=0.02907153 first=-0.03953171
post_attention len=4096 min=-1.05868697 max=2.52417803 rms=0.07445818 first=-0.08378471
block_0 len=4096 min=-6.52163029 max=20.70237732 rms=0.42629761 first=-0.02221121
```

## Independent logits oracle

`scripts/llama-logits-oracle` builds a small C++ program against the clean,
pinned llama.cpp checkout at revision
`6e6725459a892b49602b596339de4916c7c7965a`. It prints prompt token IDs and the
top ten final-position logits as decimal values plus raw IEEE-754 bits. The
fixed prompt `The capital of France is` produced identical output in two CPU
runs; the checked-in result is
`tools/llama-logits/qwen3-8b-q6-capital.golden`.

Run it from the repository root:

```sh
scripts/llama-logits-oracle \
  /disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf \
  'The capital of France is'
```

The oracle binary is built under `/disk1/tmp`, not committed. Override
`LLAMA_CPP_DIR` for another checkout and `LLAMA_CPP_BUILD_DIR` for another
build tree. On this host the wrapper selects the validated
`/disk1/tmp/llama.cpp-sm120-build`; elsewhere it falls back to the checkout's
`build` directory.

## SM120 CUDA validation

The older build detected the RTX 5060 Ti but aborted with `no kernel image is
available for execution on the device`. The replacement preserves the same
clean llama.cpp revision and uses a separate build directory:

```sh
cmake -S /disk1/github/ggml-org/llama.cpp \
  -B /disk1/tmp/llama.cpp-sm120-build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build /disk1/tmp/llama.cpp-sm120-build \
  --target llama-cli --parallel 8
```

Build environment: llama.cpp `6e6725459a892b49602b596339de4916c7c7965a`
(version 6039), CUDA 13.3.73, CMake 4.4.2, GCC/G++ 16.2.1 with NVCC host
compiler 15.3.0. With `LLAMA_LOGITS_GPU_LAYERS=99`, all 37/37 layers offload,
the model uses 5921.78 MiB of CUDA model buffers, and observed whole-device
peak memory is 6782 MiB.

Three CUDA runs produced identical raw logit bits. Their result is recorded in
`tools/llama-logits/qwen3-8b-q6-capital-sm120.golden`. CUDA and CPU select the
same top token and the same top-ten token set, but not identical logits: among
these ten tokens the largest absolute difference is about 0.565. This is much
larger than a rounding-level tolerance, so CPU is the strict Rust correctness
oracle. CUDA acceptance currently requires stable tokenization, the same
top-ten set, the same top-1 token, and absolute logit error at most 0.65 for
those shared tokens. A future intermediate-value comparison should determine
which quantized matrix kernels create the gap.

## Next rung

Extend the Rust Qwen3 forward harness in independently testable stages:

1. obtain independent intermediate values for block zero and set tolerances;
2. generalize RoPE, causal attention, and KV state to multiple positions;
3. iterate all 36 blocks without materializing dequantized matrices;
4. add final normalization/output projection and compare pinned logits.

The backend choice remains open. A maintained Rust CUDA backend must support
Qwen3 dense models and this GPU without bypassing `spm-gguf` validation.
Candle is the smallest plausible integration, but its dense quantized Qwen3
support must be proven first; mistral.rs is broader and substantially heavier.
