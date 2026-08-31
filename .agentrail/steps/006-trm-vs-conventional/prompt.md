The first head-to-head comparison: TRM's forward pass over streamed
weights against a conventional resident path. Same weights, same
inputs, same arithmetic.

This is plan item 5, deferred past the HRM rung. It is owed now: two
rungs of the ladder are built and NOTHING has yet compared the
streamed path against the conventional one it claims to replace.

BUILD

A conventional resident TRM forward. All 6,824,450 parameters loaded
into RAM as matrices, addressed randomly, using the existing
`spm_linear::resident` rather than a new matmul. `spm-trm`'s forward
is generic over `WeightStream` only, so the resident path needs its
own driver -- put it where the four-module ceiling says it goes, not
in `spm-trm` itself. There is no automated gate on crates per
component (docs/code_metrics.md), so a new sibling crate is fine and
is probably right.

MEASURE THREE CONFIGURATIONS, NOT TWO

1. Streamed from file (`spm-stream-file`) -- parameters never all
   resident.
2. Streamed from memory (`spm-stream-mem`) -- same streaming
   discipline, no IO. This isolates the cost of streaming from the
   cost of storage, and without it a file-vs-resident number conflates
   the two and cannot be attributed.
3. Conventional resident -- all weights in RAM, random access.

MEASURE TWO AXES, AND THE SECOND IS THE POINT

- Wall clock per forward.
- **Peak resident parameter bytes.** This is the thesis. The project
  exists to spend cheap sequential capacity instead of expensive
  fast memory, so a comparison reporting only time measures the axis
  the approach is expected to LOSE on and omits the one it is
  expected to win. Report both or the step has not done its job.

CORRECTNESS IS A PRECONDITION, NOT A RESULT

The resident path must agree with the streamed path bit for bit on
the real TRM checkpoint before any timing number is meaningful. A
performance comparison between two things that compute different
answers is not a comparison. Assert it, do not eyeball it.

WHAT THE ANSWER PROBABLY IS, AND WHY THAT IS FINE

docs/results.md already establishes the CPU reference is ~196x too
slow to saturate a page-cached read, so all three configurations are
expected to be compute-bound and land close together. Streaming will
likely cost little or nothing in time -- and at 7M parameters, 27 MB
resident is free on a 64 GB machine, so the memory axis will show a
win that does not yet MATTER.

Both of those are real findings and must be written as such. Do not
manufacture a win. The honest shape of the result is "streaming is
not costing us anything yet, and the memory advantage is not yet
load-bearing at this rung" -- which is exactly what a ladder is for,
and which sets up the rung where it starts to matter. State the
parameter count at which the resident path would stop fitting in a
plausible older GPU's VRAM, since that is the number the thesis
turns on.

RECORD

docs/results.md, in the established style: what was measured, the
numbers, and an explicit "what these numbers do not support"
section. Carry forward the standing caveats -- warm page cache,
scalar unoptimised engine, one machine, no IO overlap -- rather than
quietly dropping them.

DISCIPLINE

Hermetic tests. Weights stay outside the tree. No file over 1 MiB.
Every cargo call through scripts/serial.sh. Run `just check` before
committing, not after. Read gate warnings rather than letting them
scroll past -- the last step found a real bug that way.
