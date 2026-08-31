Make the `.spm` format encoding-aware, and add an f32 profile.

This is the prerequisite the importer cannot be written without, so it
is its own step rather than a preamble to a larger one.

THE PROBLEM. `spm-codec::packed_len(count)` returns
`count.div_ceil(4)` -- two bits per weight, hardcoded. Every place
that sizes a group calls it: `spm-file`'s reader and writer,
`spm-stream-groups`, `fabric-model`, `spm-quantize`'s emitter. The
format's `Encoding` discriminant exists precisely so a second encoding
can be added, but nothing consults it when computing bytes.

WORK:

1. Give `Encoding` a method returning the byte length of a group of
   `n` weights, and make every caller use it instead of `packed_len`.
   `packed_len` stays as the ternary implementation detail it always
   was, behind the encoding.

2. Add an f32 profile. TRM's checkpoint is f32 and stays f32 -- there
   is no quantization in this saga. Four bytes per weight, no packing.

   The format still requires a scale per group. For f32 the weights
   already carry their own magnitude, so write `1.0` and document that
   the field is inert for this profile rather than inventing a
   meaning for it. Do NOT special-case the scale out of the layout:
   the group structure is what makes the stream self-describing, and
   an encoding that skipped it would need its own reader.

3. Decide the group size story for f32. Ternary wanted groups because
   of shared scales; f32 does not. Pick a default and say why.

NON-NEGOTIABLE: the ternary golden fixture
(`spm-file/tests/golden/tiny.spm`) must stay byte-identical, and its
test must still pass unmodified. This is a format EXTENSION, not a
format change -- `version_major` does not move. If you find yourself
needing to bump it, stop and say so rather than regenerating the
fixture.

Testing:

- The golden fixture test passes untouched.
- Round-trip f32 weights through writer and reader exactly -- f32 is
  lossless here, so assert bit equality on the values, not a
  tolerance.
- A file with an f32 stream and a ternary stream in the same
  directory reads back correctly, proving the encoding is consulted
  per stream rather than per file.
- Group sizes that do not divide the weight count, for both
  encodings.

Gates: 25 LOC/function, 4 functions/module, 4 modules/crate, 350
LOC/file. `just check` green, `sw-checklist` 0 failed and no new
warnings (the sw-install Binary Freshness warning is expected), no
`#[allow]`.

Do NOT write the `.pt` reader or the importer in this step.
