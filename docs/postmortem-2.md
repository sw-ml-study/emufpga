# Postmortem 2: saga 2, steps 6 through 9

docs/postmortem-1.md covers the walking skeleton and the first two
rungs. This covers what happened after: the first head-to-head
comparison, two more rungs, and the encoding profile.

Five defects, two things that looked like defects and were not, and
the first real evidence that postmortem 1's prescription works.

## The defects, and what found each

| # | defect | found by | could the tests have caught it? |
| --- | --- | --- | --- |
| 10 | The encoding discriminant was written and never read | adding a second profile | **no** -- nothing had ever needed one |
| 11 | The measuring harness computed traffic at the wrong width | reading my own output | **no** -- it measured the instrument |
| 12 | `embed_tokens` needed the opposite layout | the reference comparison | yes, and it did -- loudly |
| 13 | Edits silently failing on rustfmt-reflowed source | listing functions afterwards | not a code defect |
| 14 | Nearly applying defect 8's fix where it did not apply | reading BDH's parameter shapes | **no** |

## Defect 10: a discriminant that existed, was written, and was read by nobody

`.spm` was made encoding-aware in saga 2 step 1 so a second profile
could be added later. Four steps later, adding one showed the
encoding was stamped and ignored in four separate places:

```
scripts/extract-checkpoint   widened bf16 to f32 unconditionally
spm_import::descriptors      stamped Encoding::F32 on every stream
spm_import::emit             sliced blobs at a hardcoded four bytes
spm_linear::stream::take     called the f32 codec without looking
```

The last would have decoded a bf16 stream as f32: **plausible garbage,
no error anywhere**. Half the values, wrong magnitudes, finite.

This is worse than defect 8, and the reason is worth stating exactly.
Defect 8 was a missing conversion. This was a **discriminant whose
existence advertised support that did not exist**. Every descriptor in
every `.spm` file ever written carried an encoding byte; every reader
ignored it. The format looked encoding-aware in its types, its
documentation and its wire layout, and was not encoding-aware in the
one place that decodes bytes.

The sharpest detail: the `Encoding` doc comment already warned about
this shape of bug. It says the discriminant "existed to allow a second
encoding while nothing consulted it when computing bytes", and
describes fixing exactly that for `bytes_for`. **The same sentence was
true one layer down, in the decoder, and nobody noticed** -- including
whoever wrote the warning.

### The generalisation

A field that is written and never read is worse than an absent field.
An absent field fails loudly at the point someone needs it. A written
one silently passes every round-trip test, appears in every hex dump,
and gives a reader every reason to believe the capability is there.

The fix is structural rather than a promise to be careful.
`GroupView` now carries its encoding, so a decoder is handed the
answer instead of having to look it up, and `spm-codec-any` is the
single place that maps a profile to a codec. A reader cannot pick a
codec by habit; there is nothing to pick from.

**Test for this class directly**: for every discriminant, ask what
would break if it were ignored. If nothing would, it is decoration.

## Defect 11: the instrument was wrong

`smol-xcheck` reported the traffic a forward pass moves. It computed
it as `weights * size_of::<f32>()`.

At f32 that is right. At bf16 it reports **double** the real traffic,
which would have reported the bf16 profile as saving nothing at all --
erasing the entire result of the step that introduced it.

Caught by reading the output and noticing the number had not moved
when the file size had halved. It now derives bytes from the
descriptors, which is the only source that knows the width.

### The generalisation

**The measuring instrument needs the same scrutiny as the thing it
measures, and usually gets less.** Every test in this repository
points at the engine. Nothing pointed at the harness, because a
harness feels like scaffolding rather than code.

The tell was cheap and general: a number that should have changed did
not. Any measurement that is expected to move is worth predicting
before running -- see "expectations are load-bearing" below.

## Defect 12: one tensor, two layouts, and the loud failure

SmolLM ties its embeddings. `embed_tokens` is the input lookup **and**
the output projection, and the two uses want the same row-major
layout, while `scripts/extract-checkpoint` writes column-major stream
order because that is what a streamed matmul wants.

Reading the extractor's blob as an embedding table gave cosine 0.002
at `hidden_0` -- before any layer had run.

This one is in the table as a defect but it is really a success. It
failed **immediately, loudly, and at the first comparable stage**,
because the reference comparison was built to bisect. Defect 8, the
same class of error, went undetected through an entire model and was
found only by a numerical comparison after the fact.

## Defect 13: edits that silently did not apply

At least three times across steps 8 and 9, a pattern replacement
matched nothing because `cargo fmt` had reflowed the source since the
pattern was written -- a multi-line call collapsed, a chain broken
across lines.

A replacement that matches nothing is a no-op, and **a no-op is
indistinguishable from a successful edit** unless something checks.
Each time, the following gate run failed on the unchanged code, which
is a slow and confusing way to learn that nothing happened.

The fix costs one line: assert the pattern is present before
replacing, and let it fail at the edit rather than at the gate.

It is a process defect rather than a code defect, and it has now cost
more time than any single code defect in this project.

## Defect 14: the fix that was correct and not general

Defect 8 established that PyTorch tensors need a row-major to
column-major transpose on import, and `scripts/extract-checkpoint`
does it for every 2-D tensor.

BDH stores every parameter as `(in, out)` rather than a torch
`Linear`'s `(out, in)`. Its raw row-major bytes **already are** stream
order. Running the generic extractor over it would have reintroduced
defect 8 in mirror image -- and, as defect 8 itself proved, a byte
round-trip test would not have noticed.

Caught by checking the parameter shapes against the reference forward
pass before importing, rather than reaching for the tool that worked
last time.

### The generalisation

**A conversion is a property of a pair -- the source's storage
convention and the consumer's access pattern -- not a property of a
tensor.** Three rungs give three answers for the same conversion:

| model | what the transpose should do |
| --- | --- |
| TRM, SmolLM layers | apply it |
| BDH | skip it |
| SmolLM `embed_tokens` | both, for one tensor, depending on use |

A rule learned from one framework's convention is a rule about that
convention. Postmortem 1 called this "bytes versus meaning"; this is
the same lesson arriving from a direction the rule as written did not
cover.

## Things that looked like defects and were not

Recognising these cost real time, and the recognition transfers.

**`transformers` norms only its last hidden state.** `hidden_states[30]`
is `norm(layer_30(...))` while every intermediate entry is a raw layer
output. Comparing the un-normed state against it reports cosine 0.32
and looks exactly like a broken final layer. Resolved by asking the
checkpoint -- running the reference's own `model.norm` and comparing --
rather than by reasoning about which of my layers might be wrong.

**bf16 agreeing with f32 to twelve decimal places.** The step's own
prompt predicted an accuracy cost and said that exact agreement would
mean the bf16 path was secretly still f32. It agreed anyway. The
explanation was better than the prediction: **the checkpoint is
natively bf16**, so widening to f32 and rounding back is lossless.

Confirmed rather than assumed, three ways: the file carries
discriminant 3 on disk, a directly compared layer has 0 of 576 values
differing, and a test asserts that values needing more than 8 mantissa
bits *do* come back rounded -- so if the path were secretly f32, it
fails.

### Expectations are load-bearing

Both of these were resolved quickly **because the expected answer had
been written down first**. A surprising result is only visible as a
surprise if there was a prediction to violate.

The bf16 case is the sharper one: a happy result -- no accuracy loss
-- is exactly what a broken implementation would also produce, and
without a written expectation there was no reason to look twice. The
prediction being *wrong* is what forced the verification that made the
finding solid.

Write down what the number should be before running the thing that
produces it.

## What worked, and why this document exists

Postmortem 1's actionable change was: **build the reference comparison
before the streaming path, not after.** It closed by noting that HRM,
the rung it predicted, had followed it only partly.

Steps 7 and 8 followed it properly, and the difference is not subtle:

| rung | reference read first? | bugs found | how |
| --- | --- | ---: | --- |
| TRM | no | 3 | numerical comparison, after shipping |
| HRM | partly | 1 | official implementation, after shipping |
| BDH | **yes** | **0** | nothing disagreed |
| SmolLM | **yes** | 1 | caught at `hidden_0`, before any layer ran |

**BDH verified on the first run, at cosine 1.0 for every stage, with
no bisection at all.** That is a first for this project, and it is not
luck: three details of BDH would each have been a silent bug if
guessed -- `get_freqs` quantizes indices in pairs, `.tril(diagonal=-1)`
excludes the diagonal, and `nn.LayerNorm` subtracts the mean where the
earlier rungs' norm does not. All three were read out of the reference
before any Rust was written.

That is the return on writing postmortems, stated as a measurement
rather than a hope.

## What this predicts about the next steps

1. **Every discriminant should be audited the way `Encoding` was.**
   `lane_count` and `group_size` are written into every descriptor.
   Ask what would break if each were ignored, and if the answer is
   "nothing", say so in the docs rather than leaving the field looking
   load-bearing.
2. **Ternary is the next place defect 10 can recur.** It has existed
   since saga 1, has never run against a real model, and needs a
   scale-aware accumulator that the streamed path does not have.
   `spm-codec-any` refuses it explicitly, which is the right
   behaviour, but "refuses explicitly" is not "supported".
3. **The reference-first rule now has evidence behind it.** Apply it
   to the next rung without relitigating.
4. **Predict every headline number before measuring it.** Defect 11
   and the bf16 non-defect were both caught, or made solid, by having
   an expectation to check against.
