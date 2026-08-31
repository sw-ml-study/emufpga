Build the conceptual fabric model -- the research's "virtual FPGA".
Read docs/plan.md sections 3-5 and docs/code_metrics.md first.

SCOPE, stated first because the previous version of this step got it
wrong. We do NOT want high-fidelity FPGA emulation. No LUT4 counts, no
device profiles, no "does it fit on a 9K", no place-and-route
prediction. We want something FPGA-LIKE that can be refined later: a
parameterized model whose knobs are abstract (lanes, FIFO depth, fetch
rate) rather than tied to any part.

`components/device/` stays in the tree as cited reference data with no
consumer. Do not build against it. Nothing in this step may import
`gowin-profile`.

Deliverable: `components/fabric/` plus `emufpga sim`.

The model executes a real `.spm` file and reports two things at once:

1. **Results.** The same accumulator values the CPU reference
   produces. This is what makes it a model of the engine rather than a
   spreadsheet.
2. **Cycle behaviour.** Total cycles, stall cycles, and occupancy for
   a given configuration.

Configuration knobs, all abstract:

    weight_lanes            weights the datapath consumes per cycle
    batch_width             accumulator updates per weight per cycle
    fifo_bytes              depth of the weight FIFO
    fetch_bytes_per_cycle   what the parameter store delivers per cycle
    fetch_latency_cycles    cycles before the first byte arrives

Cycles are a UNIT, not a duration. Do not convert to seconds, and do
not accept an fmax argument -- there is no sourced fmax and inventing
one is how this becomes a fit model again. Throughput questions are
answered in cycles per weight, which is refinable into wall clock the
day someone measures a clock.

The interesting output is where the pipeline stalls: whether the
datapath starves waiting on the FIFO, or the FIFO backs up waiting on
the datapath. That is the same question `eta` asks in
docs/results.md, and the two should be consistent in direction.

DIFFERENTIAL VERIFICATION IS THE POINT. The fabric must produce
accumulators identical to `spm-gemv-ref` on the same input. Note that
in column-major stream order, `weight_lanes` consecutive weights land
on `weight_lanes` DIFFERENT accumulators, so per-accumulator summation
order is unchanged and agreement should be bit-exact, not
within-tolerance. Assert bit-exact, and if it is ever not, that is a
finding worth chasing rather than a tolerance to widen. Cover
`weight_lanes` greater than the row count, where the wraparound makes
this less obvious.

Testing:

- Bit-exact agreement with the CPU reference across several
  configurations and matrix shapes.
- A starved configuration (tiny `fetch_bytes_per_cycle`) reports high
  stalls; a well-fed one reports high occupancy.
- Cycle counts are monotone in the obvious directions: more lanes
  never increases cycles; a deeper FIFO never increases stalls.
- Degenerate configs (zero lanes, zero FIFO) are refused rather than
  producing a divide-by-zero or a silently wrong count.

Gate: `just check` green, `sw-checklist` at 0 failed and no new
warnings (the sw-install Binary Freshness warning is expected), no
`#[allow]`.
