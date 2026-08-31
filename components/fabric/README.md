# components/fabric

A conceptual model of something FPGA-like. **Not an FPGA simulator.**

| Crate | Responsibility |
| --- | --- |
| `fabric-model` | abstract config, execution, cycle accounting |
| `fabric-report` | rendering a run |

## What it is not

No LUT4s, no routing, no clock domains, no packing efficiency, and
nothing that says whether a design fits a part. `components/device/`
exists in this repository and **nothing here imports it** -- a model
that could reach for LUT4 counts would drift back toward the fit
modelling this project has decided not to do yet.

## Cycles are a unit, not a duration

Nothing converts to seconds and no function takes an fmax. No fabric
clock has been measured (docs/plan.md section 6 records fmax as
`Unknown` for every board), and accepting one would turn a conceptual
model into a fidelity claim by the back door. `emufpga sim` has no
`--fmax` flag, and a test asserts it stays that way.

Throughput is reported as **cycles per weight**, which becomes wall
clock the day someone measures a clock and not before.

## The pipeline

```text
  parameter stream
        |
        v
  fetch  (fetch_bytes_per_cycle, after fetch_latency_cycles)
        |
        v
   FIFO  (fifo_bytes)
        |
        v
  issue  (weight_lanes weights per cycle)
        |
        v
 accumulate  (batch_width lanes per weight per cycle)
```

Stalls mean the datapath waited on the FIFO. Backpressure means the
fetch stage waited on the datapath. Both large means the FIFO is too
small to decouple them. This is the same question `eta` asks in
docs/results.md, and the two agree in direction.

## Correctness is not approximate

The cycle counts are a model; the arithmetic is not. Accumulators are
**bit-exact** against `spm-gemv-ref`, tested across lane counts 1 to 64
and batches 1 to 32 -- including `weight_lanes` greater than the row
count, where lanes wrap onto the next column.

That holds because the stream is column-major: `weight_lanes`
consecutive weights land on `weight_lanes` different accumulators, so
no accumulator ever sees a reordered summation. If agreement ever
becomes inexact, that is a finding to chase rather than a tolerance to
widen.

The datapath is written out here rather than shared with
`spm-gemv-ref`, so the differential test compares two implementations
of the rule instead of one implementation against itself.

Built by saga 1 step 9 (conceptual-fabric-model).
