Rung 4: SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`), a conventional
Llama-shaped transformer. The rung that decides whether this project
is a general approach or an accelerator for unusual recursive
research architectures.

WHY THIS RUNG IS ADVERSARIAL

Every model measured so far -- TRM, HRM, BDH -- is recursive. Each
re-reads one small weight set many times per forward, so the scan
amortizes itself for free. **SmolLM does not.** Thirty distinct
layers, each weight read exactly once, no rotation, no rewind. The
free lunch the first three rungs enjoyed is gone.

Shape, from the published config: hidden 576, intermediate 1536, 30
layers, 9 query heads and 3 KV heads (**grouped-query attention**,
head_dim 64), rope_theta 100000, RMS norm eps 1e-5, SwiGLU, vocab
49152, **tied embeddings**. 134,515,008 parameters.

THE QUESTION THIS RUNG MUST ANSWER

Arithmetic intensity is `batch / 4` MACs per weight-byte for ANY f32
model read once -- recursion does not change it. Work it out for TRM
and SmolLM and the answer is identical. So what does recursion
actually buy?

It buys a small **working set**, not reuse. TRM's rotating region is
27 MB; if it fits in fast memory it can be read from storage once and
re-read from cache fifteen times. This engine does not do that -- a
rewind goes back to the file -- so it pays 409 MB of traffic where 27
would do.

That has an uncomfortable implication the earlier rungs hid:
**streaming is a demonstration rather than a win whenever the working
set fits in memory anyway.** For a 27 MB rotating region on a 64 GB
machine, caching it is strictly better than streaming it. Establish
this honestly. It reframes the whole ladder and it is the strongest
argument for continuing UP it, since SmolLM read once per forward --
sequential, no re-reads, nothing to cache profitably -- is the first
workload where streaming is the right answer rather than a
demonstration.

Measure and report: traffic per forward, arithmetic intensity, and
the working set, for SmolLM beside TRM. Say plainly where each one
wins.

BUILD

A `spm-smol` crate. New things this rung needs that no earlier one
did, each of which must come from the reference and not from
assumption:

- **Grouped-query attention.** 9 query heads share 3 KV heads. TRM's
  `multi_head` is plain MHA and will not do.
- **Causal masking.** TRM and HRM see a whole puzzle at once and are
  unmasked. SmolLM is autoregressive. BDH's mask excludes the
  diagonal; a Llama causal mask includes it. Do not reuse either by
  reflex.
- **Pre-norm, not post-norm.** Llama norms the input to each sublayer.
  TRM and HRM post-norm. Getting this backwards produces plausible
  finite garbage, which is postmortem defect 7's failure mode.
- **Separate `gate_proj` and `up_proj`.** TRM fuses them into one
  `gate_up_proj`; Llama ships them as two tensors. Two streams, and
  the SwiGLU gate order must be checked against the reference rather
  than carried over.
- **Tied embeddings.** `lm_head` IS `embed_tokens`. It is 28.3 MB,
  21% of the model, and it is needed twice: as a gather at the input
  and as a 49152-row sweep at the output. Decide where it goes,
  measure the consequence, and state the reasoning -- this is the
  first rung where the resident/streamed boundary is a real trade
  rather than an obvious one.

`layouts/smollm2-135m.order`, and note that with no rotating region
the `[rotating]` section may be empty. If the order file's schema
cannot express "no rotation", that is a finding about the schema.

VERIFY

Against `transformers` on the real checkpoint, stage by stage, using
uv venv and uv pip. Weights stay outside the tree; record provenance
in docs so the next session can fetch them. The checkpoint is bf16 --
widen to f32 on import, as the TRM path already does.

Expect tolerance, not bit-exactness, against torch.

RECORD

docs/results.md in the established style with a "what these numbers do
not support" section. Update docs/plan.md if the ladder's implications
change -- and on the evidence above they probably do.

DISCIPLINE

Hermetic tests, no file over 1 MiB, every cargo call through
scripts/serial.sh, `just check` before committing, read gate warnings.
