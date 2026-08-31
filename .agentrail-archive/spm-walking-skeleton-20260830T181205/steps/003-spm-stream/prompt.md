Build `components/stream/` -- sequential parameter access, and the
metrics that make the architecture's claim measurable. Read
docs/plan.md sections 3 and 4, docs/spm-format.md, and
docs/code_metrics.md before writing code.

Remember the real complexity gates: 25 LOC per function, **4**
functions per module, **4** modules per crate (lib.rs counts as one),
350 LOC per file. Only `src/` is measured; `tests/` is not. Let the
module ceiling decide how many crates this component needs rather than
picking a count first.

Crates (adjust the split if the ceiling pushes differently):

1. `spm-stream` -- the `WeightStream` trait. Per docs/plan.md section
   3, its surface is a restriction, not a capability:

       next_block(&mut self, dst: &mut [u8]) -> Result<usize>
       rewind(&mut self)     -- between operations only

   NO seek, NO addressing, NO random access. Build on `spm-walk`'s
   `Cursor` and `spm-file`'s reader so the guarantee holds all the way
   down to the bytes.

   Include a test that demonstrates random access is not *expressible*
   through the trait -- a `tests/compile_fail/` case with
   `trybuild`, or, if you would rather not take the dependency, a
   documented test asserting the trait's method set. Say in the commit
   which you chose and why.

2. `spm-stream-mem` -- in-memory implementation, used by tests and by
   the step 004 reference engine.

3. `spm-stream-file` -- buffered file implementation. Double-buffer so
   a later async or io_uring backend slots in without changing the
   trait: while the engine consumes buffer A, buffer B fills.

4. `spm-stream-metrics` -- the numbers from docs/plan.md section 4:
   raw sequential bandwidth, decoded weights/s, useful weights/s,
   `eta` (engine consumption bandwidth / storage bandwidth), `Ps`
   (scan productivity: useful parameter applications per parameter
   value read), and `Rp` (parameter residency: parameter bytes
   resident in RAM over total parameter bytes).

   `Ps` and `Rp` are the whole economic argument, so define them
   precisely and unit-test them against hand-computed cases. At batch
   1, `Ps` must be 1.0. `Rp` must count only parameter bytes -- an
   implementation that buffers a group at a time has a tiny nonzero
   `Rp`, and the metric should show that honestly rather than
   reporting zero.

Testing:

- Known-answer tests for every metric. Hand-compute the expected value
  in the test body and say where the number comes from.
- Round-trip a `.spm` file through each implementation and assert the
  weight sequences are identical, so `spm-stream-mem` and
  `spm-stream-file` cannot diverge.
- Boundary cases: zero-weight streams, a file whose payload ends
  mid-group, `rewind` between operations.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and 0 warnings. No `#[allow]` -- fix the cause.

Do NOT implement the tensor engine, GEMV, or any device profile. That
is step 004 onward.
