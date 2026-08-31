# Postmortem 1: saga 1 and the TRM rung

What went wrong, what caught it, and what did not. Written after saga
1 (the walking skeleton) and rung 1 of the model ladder (TRM, 7M).

This is deliberately not a summary of what was built -- `plan.md` and
`results.md` cover that. It is a record of the mistakes, because they
were more instructive than the successes and several of them will
recur on the next rung if nobody writes them down.

**Continued in docs/postmortem-2.md**, which covers steps 6 through 9
-- the first head-to-head comparison, the BDH and SmolLM rungs, and
the encoding profile. It also records the return on this document:
the "read the reference first" rule below was followed properly on
BDH, which then verified on the first run with no bisection at all.

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
| 9 | HRM's input injection never performed | comparing against the official implementation | **no** |

Six of nine were invisible to a test suite that was, at the time,
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

## Addendum: the rules did not hold on the next rung

This document was written, committed, and then defect 9 landed two
commits later. That is worth recording plainly, because it says
something the rest of the document does not.

HRM's `ReasoningModule` adds an injection to its hidden state before
its layers run, and the recursion supplies a different one each time --
`z_high + input` for the low module, `z_low` for the high. The
implementation here computed `low(z_L)` where HRM computes
`low(z_L, z_H + input)`. It ran, stayed finite, produced plausible
numbers, and was not HRM. The tests checked sweep counts, rewind
counts and finiteness, and every one of them passed.

Two failures, not one:

**The actionable conclusion was written and not followed.** The last
line of this document said: build the reference comparison BEFORE the
streaming path. On the very next rung it was built second. It found
the defect on its first run, exactly as predicted, having sat unwritten
while the rest of the rung was built on top of the bug.

**"It cannot be verified" was accepted too early.** The first version
of the HRM results said the recursion could not be checked, because
`transformers` does not recognise `model_type: hrm` and the checkpoint
ships no modeling code. Both facts were true and the conclusion was
wrong: `sapientinc/HRM` *is* the modeling code, and the ported
checkpoint differs from it only in tensor names. Mike asked whether I
had considered that repository. I had read files from it and stopped at
the first obstacle -- `flash_attn` is CUDA-only -- without asking
whether the obstacle was essential. It was not: flash attention is a
performance kernel, and a stand-in on `scaled_dot_product_attention`
computes the same function.

*Rule: when concluding that something cannot be verified, name the
specific obstacle and ask whether it is essential or merely
inconvenient. "The published loader does not work" is not the same as
"the published implementation is unavailable."*

### Why a written rule was not enough

The rule was correct, prominent, and recent, and it still did not fire.
Rules that depend on remembering them at the right moment are the
weakest kind of control -- the same reasoning that put `WeightStream`'s
no-seek property in the type system and the file-size limit in a gate
rather than in a convention.

So the fix is structural, not exhortative. **A step that ports a model
does not begin until its reference comparison runs.** Written into the
step prompt as the first deliverable, with the streaming work
explicitly second. The comparison is cheap -- an afternoon on TRM, an
hour on HRM once the pattern existed -- and it has now found four
defects that nothing else did.

### The sharper statement

Defects 4, 7, 8 and 9 share one signature: *the code runs and produces
plausible output while implementing a different function than
intended.* Wrong layout, wrong normalization axis, transposed matrices,
missing injection. None crashed. None produced NaN. Every one passed a
test asserting the output was finite and had changed.

For a model port, an independent reference implementation is not
quality assurance applied after the fact. It is the **primary**
correctness mechanism, and every other test is secondary to it.

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

That last point is the actionable change for the next rung -- and it
was not followed on the rung this predicted, which is why the addendum
above exists and why the control is now structural rather than
advisory.

## Regression guards: turning the table into tests

The table above was a record. A record does not stop a defect coming
back. Every row has now been checked for whether something automated
would fail if the defect were reintroduced, and the gaps closed.

| # | defect | guard | where |
| --- | --- | --- | --- |
| 1 | `cargo fmt --workspace` | `gate.sh` runs on every check | `scripts/check` |
| 2 | stream index reported late | `visits_every_group_of_every_stream_in_order` | `spm-walk/tests/cursor.rs` |
| 3 | accumulators lane-major | **none** -- see below | benchmark only |
| 4 | alphabetical weight order | order file parsed and asserted against execution order | `spm-order/tests/no_seek.rs` |
| 5 | one projection per position | the batched form is the only one that compiles | `spm-linear` |
| 6 | intermediate width 2048 | `assert_eq!(config.intermediate(), 1536, "NOT hidden * expansion = 2048")` | `spm-trm/tests/forward.rs` |
| 7 | `rms_norm` axis | `defect_7_rms_norm_normalizes_each_position_independently` | `spm-ops/tests/regressions.rs` |
| 8 | imported matrices transposed | `defect 8: 2-D tensors are transposed into stream order` | `scripts/test-extract` |
| 9 | HRM injection never performed | `defect_9_the_input_embedding_reaches_the_recursion` | `spm-hrm/tests/regressions.rs` |

Defects 7, 8 and 9 were the ones that had no guard at all, and they
are the three that were caught only by a manual reference run needing
torch and hundreds of megabytes of weights. All three guards are now
hermetic: no checkpoint, no torch, no network. They run in the gate.

### Every guard was checked by reintroducing its defect

A guard that cannot fail is worth less than no guard, because it also
buys false confidence -- which is lesson one of this document, applied
to the fix rather than to the code. So each new guard was verified by
putting the original bug back and confirming a red test:

- Defect 7, `residual_norm` replaced with a whole-state `rms_norm`:
  `defect_7_rms_norm_normalizes_each_position_independently` FAILED.
- Defect 9, the injection term dropped from `sweep`: both HRM guards
  FAILED, reporting `changing the input embedding moved only 0 of 64
  outputs`.
- Defect 8, `to_stream_order` returning `raw` unchanged: the extractor
  guard FAILED with `got [0, 1, 10, 11, 20, 21], expected [0, 10, 20,
  1, 11, 21]`.

Then the defect was reverted and all three went green again.

The defect 7 file also carries `defect_7_a_global_norm_would_fail_the
_test_above`, which asserts the wrong implementation produces a result
the guard rejects. That one is permanent rather than a one-off manual
check: it keeps the discrimination property under test even if the
guard is later loosened.

### Defect 3 is deliberately not unit-tested

The accumulator layout is not observable through `AccumulatorBank`'s
public API -- row-major and lane-major storage compute identical
values, which is exactly why the value tests missed it. The only
difference is cache behaviour, and a throughput assertion inside a
pre-commit gate that runs under a shared build lock would be flaky in
both directions: it would fail on a loaded machine and pass on a fast
one with the bug present.

The detector for defect 3 stays the benchmark sweep, read by a person.
That is a weaker control than a test and is recorded here as such,
rather than papered over with an assertion that does not really hold.

### What runs where

The Rust guards are ordinary integration tests, so they run when their
component is gated -- which under the scoping rule means when
`components/tensor` changes. The extractor guard is different:
`scripts/extract-checkpoint` belongs to no cargo workspace, so nothing
would ever have gated it. `scripts/check` now runs `scripts/test-extract`
unconditionally, in the same always-run band as `check-size` and
`sw-checklist`, for the reason given there: a gate that goes quiet
when it has nothing to compile is a gate you stop trusting.
