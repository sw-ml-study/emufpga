# Postmortem 1: saga 1 and the TRM rung

What went wrong, what caught it, and what did not. Written after saga
1 (the walking skeleton) and rung 1 of the model ladder (TRM, 7M).

This is deliberately not a summary of what was built -- `plan.md` and
`results.md` cover that. It is a record of the mistakes, because they
were more instructive than the successes and several of them will
recur on the next rung if nobody writes them down.

## The defects, and what found each

| # | defect | found by | could the tests have caught it? |
| --- | --- | --- | --- |
| 1 | `gate.sh` passed `--workspace` to `cargo fmt` | building a throwaway crate to test the gate | no -- the gate had never run |
| 2 | Reader reported a group's stream index *after* advancing past it | extracting `spm-walk` | not as written |
| 3 | Accumulators stored lane-major; 5x throughput cliff at batch 64 | reading an anomaly in the benchmark output | no -- tests checked values, not speed |
| 4 | Weights laid out alphabetically, the reverse of execution order | reading `trm.py` after shipping | **no** -- see "layout bugs" below |
| 5 | Streaming one projection per position | writing the code that consumed it | no -- caught before it ran |
| 6 | `SwiGLU` intermediate width 2048 instead of 1536 | comparing against the reference | **no** -- see "self-confirming tests" |
| 7 | `rms_norm` normalized the whole state, not each position | bisecting against the reference | **no** |
| 8 | Every imported matrix transposed | comparing against the reference | **no** -- see "bytes versus meaning" |

Five of eight were invisible to a test suite that was, at the time,
passing 150 assertions.

## The three lessons that generalize

### Self-confirming tests

Defect 6 is the cleanest example. The `SwiGLU` intermediate width was
computed as `hidden * expansion`. The tests generated their matrix
shapes *from that same formula* and then checked the code against
them. Everything agreed perfectly, and everything was wrong.

A test that derives its expected values from the implementation
verifies self-consistency. That is worth something -- it catches
regressions -- but it cannot catch a shared misunderstanding, and it
reads exactly like a test that can.

The fix was to write the assertion against **published numbers**:
`gate_up_proj` is `(3072, 512)` in the checkpoint, so the intermediate
is 1536, and the test says so in those terms rather than in terms of a
formula.

*Rule: at least one test per component must assert against a value
that did not come from this codebase.*

### Bytes versus meaning

Defect 8 is the one worth remembering longest. The importer had a test
that wrote 15 tensors to `.spm`, read them back, and compared byte for
byte. Zero mismatches, every run. Every matrix in that file was
transposed.

The test verified **transport**. The bug was in **interpretation**.
PyTorch stores row-major; `.spm` stream order is column-major; the
bytes survived the journey perfectly and meant something different at
the far end.

Round-trip tests are cheap and worth having, but they answer "did the
bytes arrive" and are silent on "do the bytes mean what the reader
thinks". Only a computation whose answer is known independently can
answer the second.

*Rule: a format is not verified until something outside the codebase
agrees with what was computed from it.*

### Layout bugs look like working code

Defects 4, 7 and 8 share a signature: the program ran, produced finite
plausible numbers, and was wrong.

- Alphabetical stream order produced a valid file that merely could
  not be read sequentially.
- Normalizing the wrong axis produced outputs within a few percent of
  correct.
- A transposed matrix produced a cosine of 0.986 -- close enough to
  look like a precision issue.

None of them crashed, produced NaN, or tripped an assertion. A test
asserting "the state changed and is finite" passed all three, and that
test existed.

*Rule: for anything layout-shaped, assert the layout directly. "It ran
and looked reasonable" is not evidence.*

## What worked, and is worth keeping

**Bisection over end-to-end comparison.** A single end-to-end number
said cosine 0.986 and nothing more. Instrumenting each stage said: qkv
exact, attention exact, residual-norm wrong. That took one run and
pointed at a single function. The end-to-end number had been available
for an hour and had taught nothing.

**Bit-exact where bit-exact is possible.** Streamed and resident
matmuls agree to the bit, not to a tolerance, because the summation
order is identical by construction. A tolerance would have hidden a
real reordering. Against torch, exactness is impossible and a
tolerance is correct -- the distinction is worth stating explicitly
every time, because "close enough" quietly becomes the standard
otherwise.

**Constraints in the type system.** `WeightStream` has no seek, so no
consumer can express random access. `Figure` is not `Option<u32>`, so
an unsourced device figure cannot be read as a number. Neither
constraint depends on anyone remembering it.

**Recording what a result does not support.** Every measurement in
`results.md` carries its caveats in the same document, and usually in
the tool output. The importer's summary says the bytes are intact and
says nothing about inference, because at that point nothing about
inference was known.

**Reading the source instead of inferring it.** Every assumption
checked against `trm.py` and `layers.py` was cheap and settled in
minutes: RoPE theta 10000, split-half rotation, `causal=False`, no
learned norm gain. Every assumption *not* checked was a coin flip, and
two of them landed wrong.

## Process failures

**Agentrail, twice.** First, seven `--planned` entries at step 001
pre-created steps 003-009, after which every `--next-prompt` was
silently discarded -- creating a step that already exists is a no-op.
Steps 003 and 004 had empty prompts, and step 003 only got implemented
because the same session still had the text in context. Second,
`init` refuses while any saga occupies `.agentrail/`, and an attempt
to delete a step left `current_step` pointing at a removed directory.

Both were misuse, not tool defects, and the correct usage was visible
in `../sw-mlpl` the entire time. *Check a working repository before
concluding a tool is broken.*

**Scope drift, twice.** Drifted into rack architecture planning, and
separately recommended the newest GPU in the rack -- the precise
opposite of a project about cheap old hardware. Both times the user
corrected it. The goal is now restated at the top of the saga plan
where a session will read it before acting.

**A 3.1 MB session transcript nearly entered git.** `.gitignore`
covered `.agentrail/sessions/` but not the archived copy that
`agentrail archive` creates. Caught before push. The response was a
layout-independent pattern plus `scripts/check-size` as a gate,
because an ignore rule is only as good as the next `git add -f`.

**Two edits silently did not apply.** Both were pattern replacements
against source that `rustfmt` had reflowed since the pattern was
written. The script reported success; the file was unchanged. Caught
by listing the module's functions afterwards. *Verify an edit landed;
it costs one command.*

## Measuring the tools rather than trusting them

Two cases, both of which changed a decision:

`sw-checklist` warns above **4** functions per module and 4 modules per
crate. `plan.md` had been written targeting 5, which guarantees a
warning. Measuring the thresholds took one command and changed the
architecture of every component.

`compile_fail` doctests accept *any* compile error, and the
`compile_fail,E0599` annotation that should narrow it is inert under
Rust 2024's merged doctests -- verified by substituting an unrelated
type error and watching the test pass. A comment claiming the test
"fails for the right reason" would have been false. The enforced check
is now a separate file that stops compiling if the trait's required
surface changes.

## What this predicts about HRM

HRM is 27M, two modules rather than one shared block, and a different
architecture family. The failure modes above transfer directly:

1. Its intermediate width, head count and norm placement must come
   from **its** published config, not from TRM's shapes.
2. Its consumption order must be read from its forward pass, not
   guessed from tensor names.
3. Its weights need the same row-major to column-major conversion, and
   a round-trip test will again not notice if that is skipped.
4. The reference comparison should be built **before** the streaming
   path, not after. On TRM it was written last and found three bugs in
   an afternoon; every hour before that was spent on code that was
   quietly wrong.

That last point is the actionable change for the next rung.
