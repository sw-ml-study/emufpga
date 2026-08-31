Rung 3: BDH, the Dragon Hatchling (pathwaycom/bdh, arXiv 2509.26507),
streamed and verified against the official implementation.

Structure: BDH's loop body carries no layer index, so one parameter
set is applied n_layer times -- a rotating store by construction, the
TRM property arriving for free. 9 rotating streams, 5 rewinds per
forward, 604 MB of traffic for a 101 MB model. lm_head is read once
after the last level, so it sits after the rotating region and is
reached by reading on, never by seeking; asserted rather than assumed.

Verified first try: cosine 1.000000000000 at every stage, relative
2.5e-6. No bisection needed because nothing disagreed -- a first for
this project, and attributable to reading the reference forward pass
BEFORE writing code, which is the change postmortem 1 prescribed and
the HRM rung followed only partly.

THE FINDING IS NEGATIVE, which makes it worth more than the previous
two rungs' confirmations. plan.md section 3 permits resident
activations because they are kilobytes against megabytes of weights.
For BDH that inverts with sequence length: 3.2% of the weight set at
16 positions, 51.5% at 256, and 102.9% at 512 -- more activation than
model, with the crossover near 498. It is arithmetic, not incident:
the sparse latent is positions*heads*latent and grows linearly while
the weight set does not grow at all. This is after halving it by
folding relu into an in-place gate. For the Tang Nano 9K, whose BRAM
is kilobytes, that disqualifies BDH at these hyperparameters -- not
because the parameters are too big, which is the problem this project
solves, but because the state is. plan.md now records both the
specific result and the general lesson.

Second finding: the row-major-to-column-major transpose is NOT
universal. extract-checkpoint transposes because torch Linear stores
(out, in); BDH stores (in, out), so its raw bytes already are stream
order and the generic extractor would have reintroduced defect 8 in
mirror image, invisibly to a byte round-trip test.

spm-ops gained layer_norm and residual_layer_norm with tests asserting
they differ from the rms_norm pair. spm-bdh-ops holds BDH's
parameterless math including a linear attention that never
materialises its T x T scores.

Gate clean: 191 checks, 0 failed, 1 standing warning. Three complexity
violations extracted rather than allowed. Pushed as dfcbfcc.