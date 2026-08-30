# components/stream

Sequential parameter access, and the metrics that make the
architecture's claim measurable.

`WeightStream` is stated as a restriction, not a capability: two
methods, neither of which can express a position. An engine generic
over the trait has no vocabulary for random access, so the design
cannot quietly erode back into a memory controller. See
[../../docs/plan.md](../../docs/plan.md) section 3.

| Crate | Responsibility |
| --- | --- |
| `spm-stream` | the `WeightStream` trait and its error type |
| `spm-stream-mem` | in-memory implementation; the reference other backends must match |
| `spm-stream-file` | file-backed implementation over a two-slot buffer pair |
| `spm-stream-groups` | scale groups pulled off any `WeightStream` |
| `spm-stream-metrics` | bandwidth, `eta`, `Ps`, `Rp` |

## What the no-seek guarantee actually is

A guarantee about **consumers**, not implementations. A file can seek;
a slice can be indexed. What the trait ensures is that code written
against `impl WeightStream` cannot reach those operations. That is the
guarantee that matters, because implementations are few and reviewed
while consumers are many and will be written by future sessions.

Two tests hold the line. `tests/surface.rs` is the enforced one: it
implements the trait with exactly two methods, so adding a third
required method breaks the build. The `compile_fail` doctest is the
illustrative one -- it shows a seek call failing to compile, but
`compile_fail` accepts any compile error, and the error-code
annotation that would narrow it is not enforced under Rust 2024's
merged doctests.

## Metrics

`Ps` (scan productivity) and `Rp` (parameter residency) carry the
economic argument. Batch-1 dense inference gives `Ps == 1`; batching,
MoE scheduling and speculative decoding all exist to raise it.
Conventional inference sits at `Rp ~= 1`; the goal is `Rp -> 0` while
activation memory stays nonzero. Every ratio returns `Option`, because
a scan that read no weights has no scan productivity -- which is not
the same as a scan productivity of zero.

## Known limitation

`spm-stream-file` reads through two buffer slots so a prefetch thread
or io_uring backend can drop in without touching the trait, but the
refill is **synchronous today**, so IO is not yet overlapped.
Behaviourally that is a single buffer. `eta` reports the truth either
way.

Built by saga 1 step 3 (spm-stream).
