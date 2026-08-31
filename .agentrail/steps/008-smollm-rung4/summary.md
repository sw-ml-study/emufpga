Rung 4: SmolLM2-135M, the first non-recursive model on the ladder.
Verified against transformers at cosine 1.0 for every hidden state and
the logits, relative 2.4e-6, at sequence lengths 1 through 32.

30 distinct layers, each weight read once, ZERO rewinds -- the free
amortization the first three rungs enjoyed is gone, and the tests
assert its absence.

THE REFRAMING, which is the real output of this step. Arithmetic
intensity is batch/4 MACs per weight-byte for any f32 model read once.
TRM does 6,815,744*15*batch MACs on 409 MB; SmolLM does
106,168,320*batch on 425 MB. Identical. Recursion does not buy reuse,
it buys a small working set -- which means streaming is a
demonstration rather than a win whenever the working set fits in
memory anyway. TRM's 27 MB rotating region fits everywhere, so caching
beats streaming it, and the first three rungs were all in that regime.
SmolLM is the first rung that is actually the case this project is
for. plan.md now says to reason from it.

Confirmed by measurement: SmolLM demands 2585/524/143 MB/s at
positions 1/8/32 against TRM's 2854/637/166 -- the same curve for a
6.8M recursive and a 135M conventional model, exactly as batch/4
predicts.

THE COST: 21% of this model cannot be streamed. Tied embeddings put
28,311,552 of 134,515,008 weights in a table gathered by token id.
Against 0.13% for TRM and 0.05% for HRM -- a research model with a
tiny vocabulary flatters this architecture. The selecting-sweep
alternative (6.7% more traffic to remove 21% residency) is recorded as
a measured choice rather than taken silently.

Third view of the transpose: TRM needed it, BDH needed it skipped,
SmolLM needs it both ways for one tensor.

New operators with discriminating tests: GQA, a causal mask that
includes the diagonal, weighted RMS norm, pre-norm, split gate/up.

Gate clean: 201 checks, 0 failed, 1 standing warning. Pushed eb794a6.