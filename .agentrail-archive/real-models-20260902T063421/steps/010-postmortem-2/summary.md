docs/postmortem-2.md, covering saga 2 steps 6-9, plus an in-place
correction to a conclusion results.md still carried.

postmortem-1 stopped at step 5, so four steps were unrecorded.

Five defects written up: the encoding discriminant written and never
read (worse than defect 8, because a discriminant that exists
advertises support that does not -- and the Encoding doc comment
already warned about that exact shape one layer up); the measuring
harness computing traffic at the wrong width, which would have erased
the bf16 result entirely; embed_tokens needing the opposite layout;
edits silently failing on rustfmt-reflowed source at least three
times; and nearly applying defect 8's fix to BDH where it does not
apply.

Two things that looked like defects and were not got their own
section, because recognising them cost real time and both were
resolved fast only because the expected answer had been written down
first. The bf16 one is the sharper: a happy result is what a broken
implementation also produces.

The point of the document is the evidence that postmortem 1's
prescription works. TRM read no reference first and had 3 bugs found
after shipping; HRM partly, 1; BDH properly, ZERO, verified first run
with no bisection; SmolLM properly, 1 caught at hidden_0 before any
layer ran.

Also corrected results.md's 'Ps under-reports recursion by 15x' in
place. Step 8 showed arithmetic intensity is batch/4 for any f32 model
read once, so re-reading is traffic and not reuse. The old section
pointed the wrong way and sat ABOVE the section correcting it.

Guard added: declared_widths_account_for_every_payload_byte, asserting
the descriptors are the authority on width for every profile. It
needed its own builder because the existing test helper writes ternary
packing regardless of the descriptor -- defect 10's shape again, in a
test helper.

Gate clean: 211 checks, 0 failed, 1 standing warning. Pushed 0f8b75f.