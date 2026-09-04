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

## Block-zero reference

The pinned oracle also supports `LLAMA_DUMP_DIR`. Its stable llama.cpp graph
callback requests only `inp_embd`, `ffn_inp-0`, and `l_out-0`, then copies the
evaluated F32 tensors to that directory. No upstream source modification is
required. For the five-token oracle prompt, each file contains five contiguous
4096-element vectors; position zero is the first 16,384 bytes.

```sh
mkdir -p /disk1/tmp/qwen3-block0-cpu
LLAMA_DUMP_DIR=/disk1/tmp/qwen3-block0-cpu \
  scripts/llama-logits-oracle \
  /disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf \
  'The capital of France is'

cd components/cli
SPM_QWEN_REFERENCE_DIR=/disk1/tmp/qwen3-block0-cpu \
  cargo run --release -p spm-qwen3-slice -- \
  /disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf 785
```

The complete-vector comparison gives:

| Stage | Maximum absolute error | Mean absolute error | Cosine |
|---|---:|---:|---:|
| embedding | 0 | 0 | 1.000000000000 |
| post-attention | 0.003290944 | 0.000642471 | 0.999941075717 |
| block zero | 0.054967880 | 0.003724159 | 0.999939510587 |

The Rust harness enforces respective maximum-error tolerances of 0, 0.006,
and 0.06, plus cosine minima of 0.999999999, 0.99985, and 0.9997. Exact
embedding agreement rules out row orientation and Q6_K decoding errors. The
small error after several quantized matrix multiplies, and its growth through
the MLP, is consistent with different floating-point accumulation order. It
does not justify changing the model equations to imitate one CPU kernel.

Reference artifact SHA-256 values (kept outside git because the full files
include all five positions) are:

```text
inp_embd.f32  dd616ec086befd9912ce6feee9a9205701f73c307e62a6d46caae4b2ca5fb82a
ffn_inp-0.f32 4b561e783caba3a9b72b65a2a082cdc49c835cc9b537315c974f4d30d7737d9a
l_out-0.f32   7b631c7d44fe8073b571a8f9f2a20d203569b6dcc0ef086882b9e74d98112ebc
```

## Five-token attention and KV state

The slice accepts a comma-separated sequence and now implements Qwen3's
half-split RoPE (base 1,000,000), scaled dot-product attention, the 32-query to
8-key/value grouped-head mapping, causal masking, and explicit block-local K/V
state. Projection weights are dequantized one matrix at a time and shared
across all positions; no full model or second projection matrix is resident.

```sh
SPM_QWEN_REFERENCE_DIR=/disk1/tmp/qwen3-block0-cpu \
  cargo run --release -p spm-qwen3-slice -- \
  /disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf \
  785,6722,315,9625,374
```

All five embedding vectors match llama.cpp exactly. Across all positions,
post-attention maximum absolute error is at most 0.005455017 and cosine is at
least 0.999895659269. Block-zero maximum absolute error is at most 0.054967880
and cosine is at least 0.999731226156. The enforced multi-token bounds are
0.006/0.99985 for post-attention and 0.06/0.9997 for block zero. Tests cover
position-zero identity, a hand-computed RoPE pair, stable normalized softmax,
causal exclusion of a later value, and four-to-one GQA head mapping.

## Full dense-model result

The Rust harness now streams all 36 transformer blocks, retains a separate K/V
cache for each layer, applies the final RMS normalization, and evaluates the
151,936-row output projection in chunks of 256 rows. Layer projections are
decoded one at a time. The largest decoded layer matrix is about 192 MiB; the
output chunks decode to 4 MiB. Two host process samples during the release run
reported 220,628 and 220,672 KiB RSS. The five-token run took roughly 36
seconds on this host.

The llama.cpp callback can dump every `l_out-N` tensor when all token outputs
are requested. Across 180 layer-position vectors, Rust's lowest cosine was
0.997130500531. The largest absolute difference was 286.366210938 at the
position-zero outlier, where activation magnitude approaches 9,000 and cosine
remained 0.999998945652. The harness therefore enforces maximum absolute error
287 and minimum cosine 0.997 for optional all-layer reference checks.

Final Rust logits retain llama.cpp's exact top-ten token set and top token.
The largest shared-token difference against the checked-in strict CPU golden
is about 0.441. Setting `SPM_QWEN_LOGITS_GOLDEN` enforces the same top token,
the same top-ten set, and maximum absolute logit error 0.5.

## MoE follow-up

A mixture-of-experts model provides the next serial-weight experiment. For
each token and layer, the execution schedule is:

1. stream the router matrix and compute router logits;
2. select and normalize the top-k expert scores;
3. for each selected expert in stable expert-ID order, stream gate and up
   weights, compute the activation, stream down weights, and accumulate the
   score-weighted output;
4. release that expert's weights before opening the next expert;
5. compare router logits, selected IDs, individual expert contributions, the
   combined MoE output, and final logits with llama.cpp.

This keeps only one expert projection resident and needs no Python checkpoint
loader. The next rung should first select a maintained GGUF MoE under 12 GB,
pin its source hash, and establish these callback oracle points before adding
the serial Rust implementation.

The dense backend choice remains open. A maintained Rust CUDA backend must
support Qwen3 dense models and this GPU without bypassing `spm-gguf`
validation. Candle is the smallest plausible integration, but its dense
quantized Qwen3 support must be proven first; mistral.rs is broader and
substantially heavier.
