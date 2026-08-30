Build `components/cli/` -- the `emufpga` binary, starting with the
`pack` subcommand. Read docs/plan.md sections 5 and 9,
docs/spm-format.md, and docs/code_metrics.md first.

Gates: 25 LOC/function, **4** functions/module, **4** modules/crate
(lib.rs counts), 350 LOC/file. Only `src/` is measured. Let the module
ceiling decide the crate count -- components/tensor needed six crates
for what looked like four.

Deliverables:

1. `emufpga-cli` (library) and `emufpga` (thin binary). Splitting them
   keeps the subcommand logic testable: integration tests can call the
   library directly instead of shelling out.

2. `emufpga pack` -- converts a dense matrix to a ternary `.spm` in
   consumption order. Decide and DOCUMENT the input format. A plain
   text matrix is probably right for saga 1 (readable, diffable, no
   dependency); real model import is saga 4. Quantization to ternary
   and the choice of group scales must be explicit and documented --
   do NOT silently pick a rounding rule.

3. `emufpga --help` and `emufpga --version` must satisfy `sw-checklist`
   CLI validation. Run `sw-checklist -v` and read what it actually
   checks rather than guessing at the format.

Argument parsing: prefer no dependency if it stays readable; `clap` is
acceptable if hand-rolling turns into a mess. Say which you chose and
why in the commit. If you take the dependency, pin it and run
`scripts/check-locks.sh --fix`.

Testing:

- Round-trip: pack a matrix, stream it through `spm-gemv-ref`, and
  compare against `spm-numeric`'s reference on the same input. The
  packer is only correct if the engine reproduces the reference from
  its output.
- A `spm-vectors` golden case packed by the CLI must produce bytes
  identical to the case written directly through `SpmWriter`, so the
  CLI cannot drift from the library.
- Exit codes and stderr on bad input: missing file, malformed matrix,
  non-ternary values, a group size of zero.
- `--help` and `--version` output, asserted.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and 0 warnings, no `#[allow]`.

Do NOT build `bench` or `fit` yet -- those are steps 006 and 008. Keep
the subcommand dispatch open enough that adding them is not a rewrite.
