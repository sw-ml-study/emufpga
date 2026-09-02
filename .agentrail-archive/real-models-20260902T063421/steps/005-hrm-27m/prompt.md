Rung 2 of the model ladder: HRM, 27M parameters, from
`sapientinc/HRM`.

BUILD THE REFERENCE COMPARISON FIRST. This is the actionable
conclusion of docs/postmortem-1.md and it inverts how rung 1 was done.
On TRM the torch comparison was written last, found three bugs in an
afternoon, and every hour before it went into code that was quietly
wrong. Do not repeat that.

Order of work:

1. **Get HRM's config and forward pass from source**, not from
   inference. Read its published implementation the way `trm.py` and
   `models/layers.py` were read: the exact `rms_norm` axis, the
   `SwiGLU` intermediate formula, the `RoPE` base and rotation style,
   whether attention is causal, how qkv splits, and where the norms
   sit. Every one of those was a bug or a near-bug on TRM.
2. **Find a checkpoint.** `sapientinc/HRM` is the code; a checkpoint
   may or may not be published. If none exists, say so and use
   randomly initialised weights of the published shapes -- the
   streaming path does not care whether weights are trained, and rung
   2 is about mechanism.
3. **Write the torch reference and the stage dump**, mirroring
   `scratchpad/xcheck/stages.py`. Bisect from the start rather than
   reporting one end-to-end number: on TRM the single number said
   cosine 0.986 and taught nothing, while the per-stage dump pointed at
   one function in a single run.
4. **Then** the Rust side.

WHAT IS DIFFERENT ABOUT HRM, and where the risk therefore is: it has
TWO modules, an H-level and an L-level, where TRM has one shared
block. So `H_layers` is non-zero and the consumption order interleaves
two distinct weight sets. The rotating region may not be a single
contiguous run -- and if it is not, `rewind` alone cannot serve it,
because rewind returns to the start of the stream. Work out what the
recursion actually demands before choosing a layout, and if a single
rotating region cannot express it, say so plainly rather than forcing
the shape.

APPLY THE POSTMORTEM'S RULES:

- At least one test asserting against **published numbers**, not
  against values this codebase computed. HRM's parameter count and its
  matrix shapes are the obvious candidates.
- Do not trust a round-trip byte test to verify meaning. It did not
  find the transpose on TRM and it will not find one here.
- Assert layout directly. "It ran and the output is finite" passed
  three separate layout bugs on TRM.
- Weights get the same row-major to column-major conversion.
  `scripts/extract-checkpoint` already does it; confirm it applies to
  HRM's tensor shapes too, including any that are not 2-D.

Reuse rather than reimplement: `spm-linear`, `spm-ops`,
`spm-codec-dense`, `spm-import`, `spm-order` and the `.spm` format all
carry over. If `spm-ops` needs an operator HRM has and TRM did not,
add it there rather than in a model-specific crate.

Hermetic tests, weights outside the tree, gates as always. No file over
1 MiB.
