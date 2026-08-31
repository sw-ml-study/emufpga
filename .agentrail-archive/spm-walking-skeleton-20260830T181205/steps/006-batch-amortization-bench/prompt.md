Add `emufpga bench` and find the crossover where compute, not storage,
becomes the limit. This is the make-or-break measurement of saga 1 --
docs/plan.md calls it Experiment 1E, and the architecture either
becomes interesting here or it does not.

Read docs/plan.md sections 4 and 7, and docs/code_metrics.md, first.

Gates: 25 LOC/function, **4** functions/module, **4** modules/crate
(lib.rs counts), 350 LOC/file. Only `src/` is measured. Let the module
ceiling decide the crate count.

Deliverable:

    emufpga bench --input <file.spm> --batch 1,2,4,8,16,32 [--repeat N]

reporting, per batch size, every metric from docs/plan.md section 4:
raw sequential bandwidth, decoded weights/s, useful weights/s, `eta`,
`Ps`, and `Rp`. The plumbing already exists in `spm-stream-metrics`
and is already driven by `spm-gemv-ref`; this step is about running it
for real and reporting honestly.

What the measurement must show, and what it must not claim:

- `Ps` should rise linearly with batch size while
  `parameter_bytes_read` stays constant. That part is already proven
  by construction in step 004's tests, so it is a sanity check here,
  not the finding.
- The finding is the CROSSOVER: the batch size at which `eta` falls
  below 1, meaning compute stops keeping up with storage. Report it as
  a measured number with the machine it was measured on.
- Timings are wall-clock on a busy laptop. Run `--repeat` passes and
  report the spread, not just a single number. If the spread is wide
  enough that the crossover is not resolvable, SAY SO rather than
  reporting a crossover you cannot support.

Known limitation you must account for: `spm-stream-file` does not
overlap IO yet (its two buffer slots refill synchronously). So
`storage_time` and `compute_time` partition wall clock rather than
overlapping, and `eta` measures a serial pipeline. That is a real
result about this implementation, not a proxy for what an FPGA would
do -- state the distinction in the output or the docs, do not let a
reader infer the wrong thing.

Also run against both `spm-stream-mem` and `spm-stream-file`. The
memory backend gives an upper bound on storage bandwidth; the file
backend gives something closer to the real thing. Reporting both makes
`eta` interpretable.

Write results to `docs/results.md`: the numbers, the machine, the
command that produced them, and what they do and do not support.
Include the crossover if it is resolvable and the reason if not.

Testing:

- Metric wiring: a bench run over a known fixture produces the `Ps`
  and `Rp` values step 004 already pins, so `bench` cannot report
  different numbers than the engine computes.
- Argument parsing: `--batch` lists, malformed lists, empty lists.
- `bench` on a malformed or missing `.spm` fails with exit code 1.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and no NEW warnings (the `sw-install` Binary Freshness warning is
expected and is not yours to retire), no `#[allow]`.

Do NOT build `fit` or any device profile -- those are steps 007 and
008.
