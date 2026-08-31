# SmolLM2-135M reference comparison

What produced the numbers in `docs/results.md`, "SmolLM2-135M, rung 4".

## Provenance

The checkpoint is `HuggingFaceTB/SmolLM2-135M`, 269 MB of bf16
safetensors. Weights never enter this repository; fetch them outside
the tree:

```sh
curl -sLO https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
curl -sLO https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/config.json
```

## Running it

```sh
uv venv && source .venv/bin/activate && uv pip install torch transformers
python reference.py . out 8
scripts/extract-checkpoint model.safetensors extracted/
emufpga import -i extracted/ -o smollm.spm --order layouts/smollm2-135m.order
cargo run --release -p spm-smol --example smol-xcheck -- out smollm.spm extracted
```

Unlike BDH, SmolLM's tensors are ordinary torch `Linear` weights
stored `(out, in)`, so `scripts/extract-checkpoint`'s transpose is the
correct one and the normal import path applies.

## Two things that look like bugs and are not

**`transformers` norms only its last hidden state.** `hidden_states[30]`
is `norm(layer_30(...))` while the intermediate entries are raw layer
outputs. Comparing the un-normed state against it reports cosine 0.32
and looks like a broken final layer.

**`embed_tokens` needs the opposite layout to everything else.** The
extractor writes column-major stream order, which is what a streamed
matmul wants. The embedding is gathered by token id, and the tied
output projection also wants row `v` contiguous -- so the cross-check
transposes it back. Read the extractor's blob directly as an embedding
table and `hidden_0` is wrong before any layer runs.

## What is verified

Every formula, against `transformers`, at sequence lengths 1 through
32, to about 1e-6. Nothing about generation quality: there is no
tokenizer here and no decoding loop.
