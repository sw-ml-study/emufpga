# Granite MoE serial expert verification

## Pinned artifact

The first MoE target is Bartowski's GGUF conversion of IBM Granite 3.1
1B-A400M Instruct. It is small enough to leave ample space on a 16 GB GPU,
and all large tensors use the Q6_K encoding already decoded by `spm-gguf`.

| field | pinned value |
| --- | --- |
| repository | `bartowski/granite-3.1-1b-a400m-instruct-GGUF` |
| revision | `2104b63ef8d79af82a28f2df8d6191823e277c54` |
| file | `granite-3.1-1b-a400m-instruct-Q6_K.gguf` |
| bytes | 1,099,212,096 |
| SHA-256 | `4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f` |
| architecture | `granitemoe`, 24 blocks, width 1,024, expert width 512 |
| experts | 32 total, softmax top-8 active per token and block |
| quantization | Q6_K; 169 Q6_K tensors and 73 F32 tensors |
| license | Apache-2.0 |

Source repository:
<https://huggingface.co/bartowski/granite-3.1-1b-a400m-instruct-GGUF/tree/2104b63ef8d79af82a28f2df8d6191823e277c54>

Upstream model:
<https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct>

Fetch and verify it outside the repository:

```sh
scripts/fetch-granite-moe
```

GGUF is a good experiment format here because it is mmap-friendly, describes
tensor shapes and encodings, and contains no executable Python object graph.
The bounded Rust parser checks lengths and offsets before reads. This avoids
Python pickle's arbitrary-code execution risk. GGUF is not intrinsically
trustworthy, however: malformed dimensions and offsets still require strict
validation. Safetensors is the better interchange choice for original dense
training weights; `.spm` remains the intended seek-free execution layout.

## Oracle and fixed fixture

The reference is llama.cpp revision
`6e6725459a892b49602b596339de4916c7c7965a`. The local callback tool uses the
public graph callback API and does not patch that checkout. For the short
prompt `Hello`, llama.cpp tokenizes to token 8279. Capture the oracle on CPU:

```sh
mkdir -p /disk1/tmp/granite-moe-oracle-hello
LLAMA_DUMP_DIR=/disk1/tmp/granite-moe-oracle-hello \
LLAMA_LOGITS_GPU_LAYERS=0 scripts/llama-logits-oracle \
  /disk1/tmp/granite-3.1-1b-a400m-q6/granite-3.1-1b-a400m-instruct-Q6_K.gguf \
  Hello
scripts/verify-granite-moe
```

The callback observes router logits and probabilities, the complete argsort,
the selected top-8 view, normalized routing weights, gate/up activations, the
eight unweighted down-projection vectors, and the combined MoE vector. The
down tensor's selected-expert axis plus top-k IDs makes each individual
expert contribution reconstructible. llama.cpp's `ffn_moe_weighted` callback
at this pinned revision names the pre-down activation rather than the weighted
down tensor, so it is not used as evidence.

Block 0 selects experts `22, 18, 23, 3, 12, 10, 24, 16` in routing-score
order. Rust processes the set in stable expert-ID order. For each ID it reads
only that expert's Q6_K up matrix, releases it, reads gate, forms SiLU(gate) x
up, releases gate, reads down, accumulates the normalized score-weighted
result, and releases the expert before advancing. A `BTreeSet` makes the
physical order explicit. The output remains associated with routing slots so
it can be compared with llama.cpp's eight individual vectors.

## Result and tolerances

The deterministic release run completes in about 0.9 seconds on this host.
A `/proc` sampling loop observed 8,256 KiB peak RSS; this is a sampled value,
not a kernel high-water mark. The 1.02 GiB GGUF stays outside git and is read
in bounded ranges. An expert projection is about 420 KiB decoded to 2 MiB;
only one projection's weights are resident at a time.

| checkpoint | maximum absolute error | cosine |
| --- | ---: | ---: |
| embedding | 0 | 1.000000000000 |
| block-0 pre-FFN residual | 0.00944000 | 0.999960168686 |
| block-0 normalized FFN input | 0.53141451 | 0.999902246758 |
| router logits | 0.01469588 | 0.999996687088 |
| normalized top-8 weights | 0.00083908 | 0.999994309610 |
| individual down contributions | 0.33307993 | 0.999456173031 |
| combined MoE output | 0.10128140 | 0.999584454419 |

Selected expert IDs match exactly. After all 24 blocks, Rust and llama.cpp
have the same top token (444) and nine of the top ten tokens overlap. The
harness enforces exact top-1 and at least 9/10 overlap. Scalar Rust and ggml
accumulate quantized dot products and experts in different orders, so exact
logit values and complete rank identity are not expected.

The longer prompt `The capital of France is` matched 39 of 40 block-0 expert
selections. Its final slot chose expert 21 in Rust and expert 0 in llama.cpp;
those scores lie at the top-8 cutoff after the small attention arithmetic
difference. This is a useful boundary-case fixture, but not the strict routing
fixture.

## Choices, limits, and next experiments

- Q6_K was chosen over Q2_K because 1.02 GiB is already far below the 12 GB
  limit and preserves a stronger numerical oracle. The previously considered
  TensorBlock repository currently exposes only Q2_K at its latest revision.
- Granite was chosen before OLMoE 1B-7B and Qwen1.5-MoE because its 32 small
  experts make per-expert reads cheap and its llama.cpp graph points are
  observable. OLMoE is the strongest next architecture-diversity check.
- This is a host Rust correctness harness, not a CUDA Rust backend and not yet
  `.spm` execution. The next useful step is to emit router and expert groups in
  physical serial order, then run the same schedule through `WeightStream`.
- GPU offload is not required for this small scalar verification. A later
  throughput experiment can compare llama.cpp CUDA with a maintained Rust
  CUDA backend after the seek-free layout is proven.
