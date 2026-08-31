# Architecture

How the pieces fit, for someone arriving cold. The argument is in
[research.txt](research.txt), the decisions and roadmap in
[plan.md](plan.md), the wire format in [spm-format.md](spm-format.md),
and the numbers in [results.md](results.md). This page is the map
between them.

## The one idea

Conventional inference treats gigantic immutable weight matrices as
though they need random-access memory, even though the dominant
operation walks them in a fully predictable order. A Serial Parameter
Machine inverts that: weights live in cheap sequential storage,
compute moves to the stream, and only activations, accumulators and
scales stay in fast memory.

Everything here follows from one restriction:

> The tensor engine may never seek into the parameter stream while
> performing an operation.

`WeightStream` has no seek in its surface, so that is a property of
the type system rather than a discipline. Metadata and activations may
live in ordinary RAM -- the restriction is on the parameter stream
alone.

## The pipeline, end to end

```text
  text matrix
      |  emufpga pack        per-group absmean quantization
      v
   .spm file                 physical execution layout, column-major
      |
      v
  WeightStream               forward-only; no seek exists to call
      |
      v
  GroupStream                scale + packed weights, one group at a time
      |
      +----------------> spm-gemv-ref     CPU reference (the oracle)
      |
      +----------------> fabric-model     conceptual FPGA-like model
```

Both consumers read the same file and must produce the same
accumulators, bit for bit.

## The six components

Each is its own cargo workspace; there is no root `Cargo.toml`. The
horizontal spread is not taste -- `sw-checklist` allows four functions
per module and four modules per crate, so the design has to spread or
it cannot pass. Several components ended up with more crates than
first sketched for exactly that reason, and the extra crates turned out
to be real seams rather than filler.

| Component | Holds | Why it is separate |
| --- | --- | --- |
| `format` | the `.spm` container | the wire contract three implementations will share |
| `stream` | `WeightStream`, group reader, metrics | where the no-seek rule is enforced |
| `tensor` | CPU reference engine, golden vectors | the correctness oracle |
| `cli` | `pack`, `bench`, quantizer | the only crate with an external dependency |
| `device` | Gowin board profiles | reference data, **no consumer** |
| `fabric` | the conceptual model, `sim` | FPGA-like, refinable, part-agnostic |

`spm-walk`'s cursor is the clearest example of a seam that earned its
place: reader, writer and `GroupStream` all walk the same (stream,
group) sequence, and extracting it immediately exposed a bug where the
reader reported a group's stream index after advancing past it.

## The two models

They answer different questions and are deliberately not merged.

**`spm-gemv-ref`** is the oracle. Naive, unoptimised, `f64` reference
matmul beside it for comparison. Its job is to be obviously right, and
it must never be optimised into agreeing with the thing it checks.

**`fabric-model`** is the conceptual FPGA. Abstract knobs -- weight
lanes, batch width, FIFO depth, fetch rate -- tied to no part. It
reports cycles, stalls and occupancy, and its cycles are a **unit, not
a duration**: nothing converts to seconds, and no function takes an
fmax.

The two are bound by differential testing. Bit-exact, not
within-tolerance: the stream is column-major, so consecutive weights
land on different accumulators and no accumulator sees a reordered
summation. `fabric-model` writes out its own datapath rather than
sharing `spm-gemv-ref`'s, so the test compares two implementations of
the rule instead of one implementation against itself.

## The datapath

The claim the whole architecture rests on is that the inner loop needs
no multiplier. It is written to be checkable by eye, in
`spm-gemv-ref/src/datapath.rs`:

```text
code & NONZERO_BIT  == 0  ->  nothing happens
code & NEGATIVE_BIT != 0  ->  accumulator -= activation
otherwise                 ->  accumulator += activation
```

Those masks are used directly rather than decoded into a `Ternary`,
because in a fabric they are not a value at all -- they are two wires
arriving off the stream. Bit 0 is the accumulator enable, bit 1 the
add/subtract select. Group scales fold into the *activation*, once per
(group, column) pair, so no multiply appears per weight.

## What is measured, and what is not

`docs/results.md` has the numbers. The short version: the CPU
reference is compute-bound everywhere, roughly 196x too slow to
saturate even a page-cached read, and the fabric model puts the
store-bound to compute-bound crossing between 4 and 16 bytes per cycle
at 8 lanes. Two models, the same qualitative answer -- arithmetic is
the scarce resource, not bytes.

Not measured, and load-bearing: **bulk memory bandwidth and fabric
fmax are `Unknown` for every board**, and a test asserts they stay that
way. Until they are measured, nothing here can honestly predict
wall-clock throughput or whether a design fits a part, and nothing
here tries.
