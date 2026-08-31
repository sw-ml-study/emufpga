# HRM reference comparison

Runs `sapientinc/HRM`'s own `ReasoningModule` on a real checkpoint and
dumps stages, so the Rust streaming path can be checked against the
published implementation rather than against a reimplementation of it.

This found a real defect on its first run: the recursion was missing
HRM's input injection. See docs/postmortem-1.md.

## Use

```
git clone --depth 1 https://github.com/sapientinc/HRM.git hrm-src
uv venv .venv && source .venv/bin/activate
uv pip install torch einops pydantic

# Mac / any machine without CUDA: flash-attn cannot be installed, and
# models/layers.py imports it at module scope with no fallback.
mkdir -p hrm-src/shim && cp flash_attn_shim.py hrm-src/shim/flash_attn.py

cd hrm-src && python ../official.py <checkpoint-dir> <out-dir>
```

Then frame the dumped weights and compare:

```
emufpga import -i <out-dir> -o ref.spm
cargo run -p spm-hrm --example xcheck -- <out-dir> ref.spm
```

## On the stand-in

`flash_attn_shim.py` implements `flash_attn_func` with torch's
`scaled_dot_product_attention`. Flash attention is a performance kernel
computing standard scaled dot-product attention -- the substitution
changes speed, not the function. Layout is the only real difference:
flash-attn takes `[B, S, H, D]`, SDPA takes `[B, H, S, D]`.

That reasoning is sound for a forward pass and is what the cosine
1.000000000000 agreement rests on. It is **not** a substitute for
training, where flash-attn's memory behaviour is the point.

**On a Linux/NVIDIA box, install the real `flash-attn` and skip the
shim entirely** -- see docs/plan.md section 8b. Re-running the
comparison there closes the last assumption in this verification.

## Name mapping

`zbloss/HRM-sudoku-extreme` is a transformers-style port; its tensor
names differ from the official module's while the shapes are
identical. `official.py` remaps them, and the load reports zero
missing and zero unexpected keys -- which is itself evidence the
mapping is right.
