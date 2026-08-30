# components/tensor

The CPU golden reference engine. This is the oracle the saga 2 fabric
simulator and the saga 6 RTL are both checked against, so its
correctness matters more than its speed.

| Crate | Responsibility |
| --- | --- |
| `spm-accum` | accumulator banks with a batch dimension |
| `spm-activations` | resident activations, and the only multiply |
| `spm-numeric` | naive f64 reference matmul and error metrics |
| `spm-gemv-ref` | ternary GEMV over a `WeightStream` |
| `spm-vectors` | reproducible golden case generation |
| `spm-vectors-text` | golden case text format |

## The inner loop has no multiplier

`spm-gemv-ref/src/datapath.rs` is the file to read. Per weight:

```text
code & NONZERO_BIT  == 0  ->  nothing happens
code & NEGATIVE_BIT != 0  ->  accumulator -= activation
otherwise                 ->  accumulator += activation
```

The masks are used directly rather than decoded into a `Ternary`,
because in the fabric they are not a value at all -- they are two
wires arriving off the stream. Bit 0 is the accumulator enable, bit 1
the add/subtract select.

Group scales are folded into the **activation**, not applied per
weight. Crossing into a new (group, column) pair recomputes
`scale * x[j]` once per lane; with `group_size == rows` that is one
multiply per column against `rows` accumulate operations.

## Why `f32` accumulators, not `i32`

The `Ternary2F32I32` profile name says `i32`, and implementing it
showed the name had settled a question that was not actually worked
out. Scale groups run along the stream, which is column-major, so a
group's scale can vary with both output row and input column. Two ways
keep the inner loop multiplier-free: pre-scale the activation (needs
`f32` accumulation, exact) or pre-scale into fixed point (allows `i32`,
cheaper in LUTs, introduces rounding). Both are real designs and the
choice belongs to saga 2 when the fabric can measure them.

This crate takes the exact one, because an oracle that carries its own
quantization error cannot adjudicate anyone else's. No format change
was needed -- the profile discriminant is a wire value and nothing on
disk depends on accumulator width.

## Batch is reuse, not throughput

Each weight is applied to every lane before it is discarded, so `Ps`
equals the lane count. A zero weight still counts as an application:
it occupies a slot in the stream and a cycle in the engine. That keeps
`Ps` a measure of reuse rather than of sparsity, which is a separate
axis the format does not yet exploit.

Built by saga 1 step 4 (spm-tensor-ref).
