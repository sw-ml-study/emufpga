Amortize one sweep of the weights across N concurrent clients, and
show it works at all.

WHY THIS IS THE STEP THAT CAN FINALLY WIN

Nine steps have produced no positive end-to-end result, for four
reasons that this step removes together:

1. The baseline was the same scalar loop, so "streaming costs 2.5%"
   was measured where nobody deploys.
2. The working set fit in RAM every time, so there was nothing to win
   (saga 2 step 8).
3. Every measurement was one forward over a fixed prompt. Generation
   -- where the weights are re-read PER TOKEN -- is the regime the
   thesis targets and is entirely unmeasured.
4. Arithmetic intensity is `batch / 4` MACs per weight-byte, but
   "batch" so far meant positions in one prompt. **During decoding,
   batch is 1 per client.** Concurrent clients are the only place
   batch > 1 arises naturally.

The mechanism fits the architecture exactly: **the weights stream once
and serve every client; attention is per-client resident work.** That
is the asymmetry docs/plan.md section 3 rests on, finally carrying
load rather than being asserted.

THE CLAIM TO PROVE

One decode step sweeps the 210 streams ONCE and produces one token for
each of N clients. Weight bytes per generated token therefore fall as
1/N:

   1 client    212.3 MB/token
   5 clients    42.5 MB/token
  20 clients    10.6 MB/token

Success at this stage is **that it works at all**, so the bar is a
working demonstration with honest numbers, not throughput.

BUILD

A new `components/serve/` component -- this is scheduling and session
state, a different responsibility from the tensor engine, and
docs/code_metrics.md prefers promoting a cluster to a new component
over growing `tensor` further.

- A KV cache per client: `layers x 2 x context x kv_width` f32.
- Decode attention: one query position against the cached prefix.
  `spm-smol-ops` is at its module and function ceilings, so this needs
  its own home.
- A step that takes N clients each contributing one token, runs ONE
  sweep, and returns N next-token logits. The streamed matmuls batch
  across clients naturally -- `positions = N` -- because a weight is
  fetched once and applied to every client before it is discarded.
  Attention does NOT batch: each client has its own cache and its own
  position, and it carries no weights, so it is resident work.

CORRECTNESS IS THE POINT, NOT SPEED

Batching must not change any client's output. Assert that N clients
decoded together produce **bit-identical** tokens to the same clients
decoded alone. This is the same discipline as every earlier rung and
it is what separates a demonstration from a plausible-looking one.

Also assert the amortization structurally: weight bytes read per
decode step must be **independent of N**. Measure it from the stream,
not from arithmetic -- postmortem 2 defect 11 is exactly the mistake
of computing a traffic figure from an assumption.

REPORT HONESTLY, INCLUDING THE COST

The KV cache is the bill. At 512 context it is 23.6 MB per client, so
five clients hold 118 MB against 212 MB of weights. That is the BDH
lesson recurring: what streaming saves on weights, serving spends on
state. Report both, and say where the crossover is.

Do NOT claim a throughput win. The engine is still scalar and ~196x
too slow; what this step establishes is that the weight traffic
amortizes, which is a property of the schedule rather than of the
arithmetic.

VERIFY AGAINST THE REFERENCE

Greedy decoding from a fixed prompt must match `transformers` greedy
decoding for the same prompt, token for token, for at least a few
tokens. Formulas were verified in step 8; what is new here is the KV
cache and the position handling, and an off-by-one in either produces
fluent-looking wrong output.

DISCIPLINE

Hermetic tests, weights outside the tree, no file over 1 MiB, cargo
through scripts/serial.sh, `just check` before committing. Assert an
edit's pattern matched before replacing it -- postmortem 2 defect 13.
