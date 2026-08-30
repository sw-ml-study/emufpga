Build `components/tensor/` -- the CPU golden reference engine. This is
the model every later implementation is checked against: the fabric
simulator in saga 2, and hand-written RTL in saga 6. Its correctness
matters more than its speed.

Read docs/plan.md sections 2-4, docs/spm-format.md, and
docs/code_metrics.md first.

Gates: 25 LOC/function, **4** functions/module, **4** modules/crate
(lib.rs counts), 350 LOC/file. Only `src/` is measured. Let the module
ceiling decide the crate count.

Crates (adjust if the ceiling pushes differently):

1. `spm-accum` -- accumulator banks with a batch dimension. `rows`
   accumulators per batch lane, `i32` per the Ternary2F32I32 profile.
   The batch dimension is the whole point: one weight arriving off the
   stream is applied to every lane before it is discarded, which is
   what raises `Ps` from 1 to the batch size.

2. `spm-gemv-ref` -- ternary GEMV consuming a
   `spm_stream_groups::GroupStream`. **No multiplier in the inner
   loop.** A weight is add, subtract, or skip:

       Plus  => acc += activation
       Minus => acc -= activation
       Zero  => nothing

   Use `spm_codec::NONZERO_BIT` and `NEGATIVE_BIT` directly rather than
   matching on the enum where it reads naturally -- the point is that
   these two bits ARE the control lines, and the reference should make
   that visible to whoever writes the RTL.

   Consumption order is column-major (docs/spm-format.md): hold
   activation `x[j]` while a whole column streams past. Apply the
   group scale to the activation, not to each weight, so the inner
   loop stays multiplier-free.

3. `spm-numeric` -- an f32 reference matmul and the error metrics from
   docs/plan.md section 4: max absolute error, mean error, cosine
   similarity.

4. `spm-vectors` -- golden vector generation and serialization. These
   files are the regression suite later RTL is validated against, so
   they must be reproducible from a seed and stable across runs.

Testing:

- Ternary GEMV must match the f32 reference within stated tolerances.
  State the tolerance and say where the number comes from; do not
  tune it until the test passes.
- Feed the SAME `.spm` file through `spm-stream-mem` and
  `spm-stream-file` and assert identical accumulator output, so the
  backends cannot diverge.
- Batch invariance: batch 1 and batch N must produce identical
  per-lane results when every lane holds the same activations. If they
  differ, the batch dimension is wrong.
- Populate a `ScanMetrics` from a real run and assert `Ps == 1.0` at
  batch 1 and `Ps == N` at batch N. This is the first time those
  metrics are driven by an actual engine rather than hand-built
  counters, so it is the first real check that the accounting is
  wired correctly.
- Boundary cases: zero-weight streams, a single-element matrix, a
  group size that does not divide the matrix.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and 0 warnings, no `#[allow]`.

Do NOT build the CLI, the fit model, or any device profile. Those are
steps 005 and 007-008.
