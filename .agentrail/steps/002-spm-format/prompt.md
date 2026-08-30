Build `components/format/` -- the `.spm` container. Read docs/plan.md
sections 3 and 5, and docs/code_metrics.md, before writing code.

`.spm` is a PHYSICAL EXECUTION LAYOUT, not a model interchange format.
Weights are stored in exactly the order the tensor engine consumes
them, so that opening a stream and reading it to the end IS the matrix
operation. Every design choice should be judged against that sentence.

Create the workspace (edition 2024, `[workspace.package]` with license
= "MIT" and version, `[workspace.lints.clippy] pedantic = "warn"`),
then run `scripts/check-locks.sh --fix` to generate its Cargo.lock --
`gate.sh` refuses a workspace whose lock is missing or stale.

Crates:

1. `spm-header` -- magic bytes, format version, endianness declaration,
   model metadata. Must be forward-checkable: a reader given a newer
   major version fails with a clear error rather than misparsing.

2. `spm-codec` -- the ternary packing codec. {-1, 0, +1} in 2 bits per
   weight with one reserved code; per-group scale factors. Decide and
   DOCUMENT the group size and the reserved code's meaning. Encoding
   and decoding must be exact inverses over every input.

3. `spm-layout` -- the stream directory and operation descriptors:
   matrix dimensions, quantization, scale layout, lane count, output
   accumulator format. This crate owns consumption-order tiling -- the
   mapping from logical (row, col) to stream position. Per docs/plan.md
   section 2, weights for a matrix-vector product are laid out so an
   activation x[j] is held while a whole column of weights streams past.

4. `spm-file` -- reader and writer composing the three above. The
   reader must expose only sequential access; do NOT add a seek or
   index-by-position method, even a private one, because step 003
   builds the no-seek WeightStream on top of this and the guarantee has
   to hold all the way down.

Testing:

- Round-trip property tests: random matrices across several shapes and
  group sizes encode then decode to the original ternary values
  exactly.
- Reserved-code and boundary cases: group sizes that do not divide the
  matrix dimension evenly, zero-length streams, single-element streams.
- A GOLDEN FIXTURE pinning the on-disk byte layout. Commit a small
  .spm file and assert the writer reproduces it byte for byte. This is
  the contract the RP2350 and rack-Linux fronts, and later the RTL,
  will all read; if it changes silently, three implementations diverge
  without anyone noticing. Note .gitattributes already marks *.spm
  binary.
- Document the byte layout in docs/spm-format.md as you go, rather
  than reconstructing it at step 009.

Gate before committing: `just check` must be green, which includes
clippy -D warnings, and `sw-checklist` must stay at 0 failed and 0
warnings -- this is the first real crate, so the zero baseline is
established here. Keep functions at or under 25 LOC and modules at or
under 5 functions from the start; retrofitting is far more expensive.

Do NOT implement WeightStream, the tensor engine, or any device
profile in this step.
