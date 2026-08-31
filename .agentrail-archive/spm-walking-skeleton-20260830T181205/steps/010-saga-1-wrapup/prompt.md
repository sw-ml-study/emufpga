Close saga 1. Read docs/plan.md end to end first -- this step is
partly about checking the plan still describes the repository.

Deliverables:

1. `docs/architecture.md` -- how the pieces fit, for someone arriving
   cold. The abstract machine, the six components and why the module
   ceiling shaped them, and the two models (CPU reference, conceptual
   fabric) with what each is good for. Short. It should not repeat
   docs/plan.md, docs/spm-format.md or the component READMEs; it
   should point at them.

2. `README.md` results table -- the headline numbers from
   docs/results.md, with the caveats that make them readable, and a
   link. A reader should be able to tell in thirty seconds what was
   measured and what it does not claim.

3. Verify docs/plan.md against the tree. Sections 5 and 8 have been
   edited step by step and are the most likely to have drifted. Fix
   what is stale. If a section is now wrong in a way worth explaining,
   explain it rather than quietly rewriting it.

4. Final `sw-checklist` counts recorded, and a note on the one
   standing exception (Binary Freshness / sw-install, which the
   checkpoint process forbids running unasked).

5. Define saga 2 in docs/plan.md section 7 with enough detail to start
   from, informed by what saga 1 actually measured rather than by what
   it was expected to measure. Candidates, in the order the evidence
   supports them:

   - overlapped fetch in `spm-stream-file`, so `eta` stops measuring a
     serial pipeline;
   - Q4 and bitplane weight layouts alongside ternary, so conclusions
     stop applying only to BitNet-style models;
   - a real tiny model (TRM) instead of synthetic matrices.

   Pick an order and say why. Do not commit to hardware work: the two
   figures that would justify it -- bulk memory bandwidth and fabric
   fmax -- are still Unknown.

6. `agentrail complete --done`.

Gate: `just check` green, `sw-markdown-checker` clean on every
hand-written `.md`, `sw-checklist` at 0 failed and no new warnings.

Write no new production code in this step. If something needs code,
that is saga 2.
