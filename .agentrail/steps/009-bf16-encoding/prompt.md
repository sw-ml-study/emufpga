Halve the traffic: a bf16 encoding profile, end to end, on a real
model.

WHY THIS AND NOT ANOTHER RUNG

Four rungs are verified and every one of them streams f32. But
SmolLM2-135M's checkpoint IS bf16, and `scripts/extract-checkpoint`
widens it on the way in -- so the .spm is 538 MB where 269 would do,
and every traffic and bandwidth number in docs/results.md is twice
what it needs to be. That is a self-inflicted 2x on the one axis this
project exists to improve.

`.spm` was made encoding-aware in saga 2 step 1 precisely so a second
profile could be added. Nothing has exercised that. Adding one is
worth more right now than a fifth model.

WHAT IS ACTUALLY HARDCODED

The encoding is stamped and ignored in three separate places, and the
`Encoding` doc comment already warns about exactly this failure:

1. `scripts/extract-checkpoint` widens bf16 to f32 unconditionally.
2. `spm_import::descriptors` stamps `Encoding::F32` on every stream,
   ignoring the dtype the manifest already carries.
3. `spm_linear::stream::take` calls `spm_codec_dense::decode_into`
   without consulting the descriptor at all -- so a bf16 stream would
   be silently misread as f32, producing plausible garbage rather than
   an error.

Point 3 is the dangerous one and it is the postmortem's failure mode
again: a discriminant that exists, is written, and is never read. Fix
it so a mis-encoded stream cannot be decoded by the wrong codec.
`GroupView` knows its stream index and the descriptors are right
there; carry the encoding on the view rather than making every caller
look it up.

BUILD

- `Encoding::Bf16`, discriminant 3, `bytes_for(n) = 2n`.
- A `spm-codec-bf16` crate beside `spm-codec-dense`. bf16 is the top
  16 bits of an f32, so decoding is a shift and encoding is a
  round-to-nearest-even truncation -- write the rounding out, do not
  just drop the low bits, and test the halfway cases.
- `extract-checkpoint` gains a flag to emit bf16 blobs rather than
  widening, and records the dtype so the importer can act on it.
- `spm_import::descriptors` reads the manifest dtype.
- The reader dispatches on the descriptor's encoding.

VERIFY

Three comparisons, and all three matter:

1. bf16 `.spm` against the f32 `.spm` on the same model -- these
   should differ by roughly bf16's 8-bit mantissa, NOT agree exactly.
   Exact agreement would mean the bf16 path is secretly still f32.
2. bf16 against `transformers` -- the accuracy actually paid.
3. The file is half the size and the traffic is half. Measure, do not
   assume.

Report the accuracy cost honestly. bf16 has 8 mantissa bits against
f32's 24, so a cosine of 1.0 to twelve places is NOT the expected
answer and reporting one would mean something is wrong.

RECORD

docs/results.md, with what the numbers do not support. If halving
traffic changes what the ladder implies about demanded store
bandwidth, say so.

DISCIPLINE

Hermetic tests, weights outside the tree, no file over 1 MiB, cargo
through scripts/serial.sh, `just check` before committing. Watch the
complexity gates: several of these crates are at their module or
function ceilings already.
