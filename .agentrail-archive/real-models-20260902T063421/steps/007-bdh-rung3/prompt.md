Rung 3 of the model ladder: BDH, the Dragon Hatchling, from
`pathwaycom/bdh` (Kosowski, Uznanski, Chorowski, Stamirowska,
Bartoszkiewicz, arXiv 2509.26507).

READ FIRST, BUILD SECOND. The reference forward pass has already been
read for this prompt; the facts below come from it, not from guessing
at tensor names. That ordering is the standing rule from
docs/postmortem-1.md and it is not optional.

WHAT BDH IS, STRUCTURALLY

Defaults: n_layer 6, D 256, nh 4, mlp_internal_dim_multiplier 128,
vocab 256, so N = 128 * 256 / 4 = 8192.

  decoder     (nh*N, D)   8,388,608
  encoder     (nh, D, N)  8,388,608
  encoder_v   (nh, D, N)  8,388,608
  lm_head     (D, vocab)     65,536
  embed       (vocab, D)     65,536
                          25,296,896 = 101.2 MB f32

The forward loop is:

  for level in range(n_layer):
      x_sparse = relu(x @ encoder)
      yKV      = ln(attn(Q=x_sparse, K=x_sparse, V=x))
      y_sparse = relu(yKV @ encoder_v)
      xy       = x_sparse * y_sparse
      y        = ln(reshape(xy) @ decoder)
      x        = ln(x + y)
  logits = x @ lm_head

**The loop body carries no layer index.** BDH applies ONE parameter
set n_layer times, so it is a rotating store by construction, like
TRM. Consumption order per level is encoder -> encoder_v -> decoder,
and `lm_head` is read exactly once after the last level -- so it
belongs at the END of the layout, after the rotating region, and is
reached by simply continuing to read. That is the same shape as HRM's
[low][high]: verify it rather than assuming it.

Attention carries NO learned parameters. RoPE frequencies are a
computed buffer, and `scores = (QR @ KR.mT).tril(diagonal=-1)` is
strictly lower triangular -- note the -1, which excludes the diagonal,
unlike a conventional causal mask.

THE FINDING THIS RUNG EXISTS TO PRODUCE

`x_sparse` must survive until after `encoder_v` has been consumed,
because `xy = x_sparse * y_sparse`. Both are nh*T*N floats:

  T=16    4.2 MB      T=64    16.8 MB
  T=32    8.4 MB      T=256   67.1 MB   (66% of the whole weight set)

docs/plan.md section 3 justifies resident activations on the grounds
that they are kilobytes while weights are megabytes. **For BDH at any
useful sequence length that is false**, and it gets worse linearly in
T. Measure it, state it plainly, and say what it means for the FPGA
target where BRAM is the scarce resource. Do not soften it: a rung
that contradicts an assumption is worth more than one that confirms
it, and this is the first architecture on the ladder that does.

BUILD

A `spm-bdh` crate streaming the rotating region, with the same
discipline as `spm-trm` and `spm-hrm`: no seek, rewind between levels
only, activations resident. A `layouts/bdh.order` file. Reuse
`spm-ops` where the operators already exist and add to it only what
BDH genuinely needs (relu, elementwise product, the tril-with-
diagonal-excluded linear attention).

Note that `embed` is a gather by token id, not a sweep -- a lookup
table cannot be streamed to serve one token without reading all of
it. At 256 KB it stays resident, and that is a legitimate instance of
plan section 3 rather than a violation. Say so explicitly rather than
letting it pass unremarked.

VERIFY AGAINST THE REFERENCE

`pathwaycom/bdh` ships architecture and training code; if it ships no
trained checkpoint, verify against the official module on SEEDED
RANDOM weights exported from torch. That still checks every formula
exactly, which is what the TRM and HRM cross-checks actually caught
bugs with. Use uv venv and uv pip, never pip directly.

Expect tolerance rather than bit-exactness against torch, and
bit-exactness only between this repo's own streamed and resident
paths.

RECORD

docs/results.md in the established style, including a "what these
numbers do not support" section. If BDH changes what the ladder
implies, say so in docs/plan.md too.

DISCIPLINE

Hermetic tests, weights outside the tree, no file over 1 MiB, every
cargo call through scripts/serial.sh, `just check` before committing.
