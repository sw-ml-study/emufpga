Build the resource budget and `emufpga fit`. Read docs/plan.md
sections 5-7, docs/results.md, and docs/code_metrics.md first.

Gates: 25 LOC/function, **4** functions/module, **4** modules/crate
(lib.rs counts), 350 LOC/file. Only `src/` is measured.

Deliverable:

    emufpga fit --lanes N --batch N [--group-size N]

reporting, per board: LUT4 / flip-flop / B-SRAM / DSP utilization for
the proposed engine, whether it fits, and the predicted cycles and
throughput.

THE HARD CONSTRAINT: `gowin-profile` records bulk memory bandwidth and
fabric fmax as `Unknown` for every board, and a test asserts they stay
that way. `fit` therefore CANNOT report a wall-clock throughput
without inventing one of them.

Do not invent it. The right shape is:

- Report what IS computable from sourced figures: resource
  utilization, and cycles per operation, which follow from the engine
  configuration and the fabric counts alone.
- Report throughput as a FUNCTION of the unknown, not a number. For
  example "tokens/s = f(fmax)" with the coefficient given, so a reader
  who later measures fmax can finish the calculation, and so saga 6
  can contradict the coefficient specifically.
- Refuse, loudly, to print a figure that depends on an `Unknown`.
  `Figure::value()` returns `Option` precisely so this is not
  accidental.

Carry step 006's measurement in: the CPU reference is ~196x too slow
to saturate a page-cached read. State, per board, how many parallel
ternary lanes would be needed to close that gap at a given assumed
fmax -- expressed as a formula plus a worked example at one clearly
labelled ASSUMED clock, never as a prediction.

Write `docs/fit-model.md` stating EVERY assumption the model makes:
LUT4 per ternary lane, flip-flops per accumulator, how B-SRAM blocks
map to activation and accumulator storage, what is ignored (routing,
IO, clock trees, packing efficiency). The point of that document is
that saga 6's real place-and-route can contradict specific claims
rather than shrug at the whole model. An assumption you cannot state
precisely is one you should not encode.

Testing:

- A configuration that exceeds a board reports "does not fit", naming
  the resource that ran out.
- The 9K's sourced figures produce a utilization; boards with unknown
  fabric fields report what they can and name what they cannot.
- Any path that would need bandwidth or fmax returns the unknown
  rather than a number.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and no new warnings, no `#[allow]`.
