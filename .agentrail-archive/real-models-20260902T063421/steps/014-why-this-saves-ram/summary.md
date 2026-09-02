docs/why-this-saves-ram.md, plus a README that leads with the goal.

Mike said we should not give up too quickly and asked for the plain
version. Re-reading, several documents hedged in a way that read as
doubt about the thesis when the measurements support it -- they are
quiet about speed, which was never the claim. Restated the goal: doing
more with less at equal correctness, not going faster.

Collected what is already demonstrated, which is more than the docs
implied: 269 MB on disk against 4 KiB resident (65,674x), bit-exact
against a resident path, 2.5% wall-clock cost, five clients with five
different questions off one sweep, bf16 halving traffic for free, four
architectures verified.

The new analysis is MoE, which nothing in docs/ covered. Its defining
problem is a VRAM problem: data-dependent routing forces every expert
resident though almost none fire. Streamed, each expert is a stream.
With E experts and top-k routing, N requests want up to N*k experts,
so past N = E/k a full sweep is FEWER bytes than fetching per request
-- 32 requests at DeepSeek's shape. Concurrency makes it better, which
is backwards from conventional serving and is the argument. Experts
are independent so device-per-expert shares nothing but activations.

The honest limit is the KV cache: 1.44 GB for five clients at 4k
against ~0 resident weights on a 300 GB model, and roughly 200x
smaller than the weights.

Listed what is not true -- no MoE run, no ternary run, no disk
measured, no energy measured, 135M is the largest -- and ordered the
next experiments cheapest first: cold store, ternary, one MoE layer.