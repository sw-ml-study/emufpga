Write docs/postmortem-2.md, covering saga 2 steps 6 through 9, and
correct a conclusion in docs/results.md that a later step superseded.

WHY NOW

docs/postmortem-1.md stops at step 5. Four steps of findings are
unrecorded, including the most serious defect the project has produced
and -- more usefully -- the first hard evidence that postmortem 1's
own prescription works.

WHAT TO COVER

Defects, with what caught each and whether a test could have:

- **The encoding discriminant was written and never read.** Step 9.
  `spm_linear::take` called the f32 codec without consulting the
  descriptor, so a bf16 stream would have decoded as f32 into
  plausible garbage with no error anywhere. The `Encoding` doc comment
  already warned about this exact shape and it was still true one
  layer down. Worse than defect 8, because the discriminant's
  existence made the format LOOK like it supported profiles.
- **The measuring harness was wrong.** Step 9. `smol-xcheck` computed
  traffic as weights x 4, which would have reported bf16's traffic as
  double and hidden the entire result. A bug in the instrument, not
  the thing measured.
- **The transpose is not universal.** Steps 7 and 8. TRM needed it,
  BDH needed it SKIPPED, SmolLM needs it BOTH WAYS for one tensor.
  Defect 8's fix was correct and not general, and applying it by
  reflex to BDH would have reintroduced it in mirror image.
- **Edits silently failing on rustfmt-reflowed source.** Recurring,
  at least three times across steps 8 and 9. A pattern replacement
  that does not match is a no-op, and a no-op looks exactly like a
  successful edit. This is a process defect and it has now cost more
  than any single code defect.

THINGS THAT LOOKED LIKE DEFECTS AND WERE NOT

Give these their own section; recognising them cost real time and the
recognition is transferable.

- `transformers` norms only its LAST hidden state, so comparing the
  raw state against `hidden_30` reports cosine 0.32 and looks like a
  broken final layer.
- The bf16 path agreeing with f32 to twelve places looked like the
  encoding was not engaged. It was losslessness: the checkpoint is
  natively bf16.

Both were resolved by checking against the source of truth rather than
by reasoning about the symptom.

WHAT WORKED, AND IS THE POINT OF WRITING THIS

Postmortem 1's actionable change was "build the reference comparison
BEFORE the streaming path". Steps 7 and 8 followed it. **BDH verified
on the first run with no bisection at all** -- a first for this
project -- and SmolLM needed one fix, found at `hidden_0` before any
layer had run. Compare TRM, where the same class of work found three
bugs in an afternoon after the fact. Say this plainly: it is evidence
that the prescription is right, and it is the reason to keep writing
these.

A CONCLUSION THAT MUST BE CORRECTED, NOT JUST SUPERSEDED

docs/results.md still carries "`Ps` under-reports recursion by 15x"
from step 4, which frames a recursive model's re-reads as reuse. Step
8 showed arithmetic intensity is `batch / 4` MACs per weight-byte for
any f32 model read once, recursion or not -- so recursion buys a small
working set, NOT reuse. The old section is not merely incomplete, it
points the wrong way, and it sits above the section that corrects it.

Annotate it in place. A reader who stops halfway down the document
must not be left with the superseded claim.

GUARDS

Add one where it is cheap and real. Defect 10 is already guarded by
`spm-linear/tests/encodings.rs`. For the harness bug, a test that the
descriptors' declared bytes sum to the payload actually present would
catch a traffic figure computed from the wrong width.

DISCIPLINE

ASCII markdown, `just check` before committing, no file over 1 MiB.
