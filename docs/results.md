# Measured results

Saga 1 step 6, the batch-amortization sweep. docs/plan.md calls this
Experiment 1E and the make-or-break measurement of saga 1.

## What was measured

```
just bench          # or: scripts/bench [rows] [cols] [group_size]
```

which is equivalent to:

```
emufpga pack  -i bench.txt -o bench.spm -g 1024
emufpga bench -i bench.spm -b 1,2,4,8,16,32,64,128 -r 9
```

The matrix is generated from a fixed seed, so a rerun on another
machine sees the same weights.

A 1024 x 512 ternary matrix: 524,288 weights, 133,184 bytes packed,
scale group 1024 (two groups per column). Weights drawn from a
standard normal and quantized with per-group absmean.

Machine: Apple M1 Max, 10 cores, macOS 26.5. Release build. Nine
passes per point, fastest reported, spread shown.

## The numbers

| backend | batch | store MB/s | useful Mw/s | eta | Ps | Rp | spread % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory | 1 | 9424 | 149.3 | 0.004 | 1 | 0.001953 | 61.3 |
| memory | 2 | 10558 | 317.6 | 0.004 | 2 | 0.001953 | 11.2 |
| memory | 4 | 11235 | 552.3 | 0.003 | 4 | 0.001953 | 4.4 |
| memory | 8 | 10956 | 959.5 | 0.003 | 8 | 0.001953 | 3.5 |
| memory | 16 | 10989 | 1502.7 | 0.002 | 16 | 0.001953 | 2.5 |
| memory | 32 | 11404 | 2621.5 | 0.002 | 32 | 0.001953 | 3.9 |
| memory | 64 | 9708 | 4897.3 | 0.002 | 64 | 0.001953 | 3.9 |
| memory | 128 | 9767 | 7396.0 | 0.001 | 128 | 0.001953 | 3.0 |
| file | 1 | 8192 | 167.2 | 0.005 | 1 | 0.001953 | 3.5 |
| file | 2 | 7865 | 318.8 | 0.005 | 2 | 0.001953 | 1.2 |
| file | 4 | 8810 | 554.1 | 0.004 | 4 | 0.001953 | 5.5 |
| file | 8 | 8548 | 958.9 | 0.004 | 8 | 0.001953 | 3.6 |
| file | 16 | 8167 | 1487.0 | 0.003 | 16 | 0.001953 | 1.8 |
| file | 32 | 7634 | 2592.7 | 0.003 | 32 | 0.001953 | 4.4 |
| file | 64 | 7769 | 4883.6 | 0.002 | 64 | 0.001953 | 3.9 |
| file | 128 | 7366 | 7395.3 | 0.002 | 128 | 0.001953 | 2.2 |

Timestamp pair overhead: 40 ns, charged to every scale group. With 512
groups that is 20 us per scan against 3 ms of work at batch 1, so the
storage/compute split is resolvable.

## Finding 1: the crossover was NOT measured

The step set out to find the batch size where `eta` falls below 1 and
compute stops keeping up with storage. **There is no such point in
this range.** `eta` is 0.005 at batch 1 and falls from there. The CPU
reference engine is compute-bound everywhere, and the crossing lies
below the smallest batch size, unmeasured.

Reporting this as "crossover at batch 1" would claim a measurement that
was never made, so `spm-bench` distinguishes the three cases in its
type system -- `AlreadyBelow`, `At`, `NotReached` -- and prints
`NOT MEASURED` here.

The useful form of the result is the ratio. At batch 1 the file store
delivered 8192 MB/s while the engine consumed packed weights at
167.2 Mw/s, which at four weights per byte is 41.8 MB/s:

```
8192 / 41.8 ~= 196
```

**The tensor engine is roughly 196x too slow to saturate even a
page-cached file read.** That is the concrete design target this step
produces, and step 008's fit model should be held to it: a fabric
configuration that does not close a ~200x gap will not be
storage-bound either.

## Finding 2: batching raises useful work and LOWERS consumption rate

Both are true at once, and confusing them would be easy.

| batch | aggregate Mw/s | per-lane Mw/s | stream consumption MB/s |
| ---: | ---: | ---: | ---: |
| 1 | 167.2 | 167.2 | 41.8 |
| 8 | 958.9 | 119.9 | 30.0 |
| 32 | 2592.7 | 81.0 | 20.3 |
| 128 | 7395.3 | 57.8 | 14.4 |

128x the batch buys 44x the aggregate throughput -- real, and the
architecture's economic argument holding up. But each weight now takes
longer to process because it is applied to more lanes, so the rate at
which bytes are pulled off the store *falls* by 2.9x.

For a compute-bound machine, which this is, batching is the right move:
it maximises useful work per byte. For a machine trying to become
storage-bound it moves the wrong way. Both regimes are in the research;
this measurement says which one the CPU reference is in.

There is no sharp knee. Per-lane efficiency declines monotonically from
batch 1 to 128, so the batch size to choose is set by latency tolerance
and accumulator memory, not by a cliff in this curve.

## Finding 3: `Ps` is exact and storage traffic is flat

`Ps` equals the batch size exactly at every point, and
`parameter_bytes_read` is 133,120 bytes at every point regardless of
batch. Reuse rises; storage traffic does not move. This was already
true by construction in step 004's tests; measuring it on a real scan
confirms the accounting is wired to the engine.

`Rp` is 0.001953 (one 256-byte group buffer against 131,072 bytes of
packed weights) and does not change with batch size. Reuse is free in
memory terms.

## Finding 4: the accumulator layout mattered by 5x

The first run showed useful throughput collapsing from 2113 to 434 Mw/s
between batch 32 and batch 64 -- worse absolute throughput for twice
the work.

Cause: `spm-accum` stored accumulators lane-major, so applying one
weight to every lane strided by `rows` -- 4 KiB apart for this matrix,
one cache line per lane. At batch 64 the 256 KiB working set passed a
64 KiB L1 and the prefetcher had nothing to work with.

Storing row-major, so a row's lanes are adjacent, removed the cliff
entirely: batch 64 now reports 4883 Mw/s and scaling stays monotone
through 128. The fix also aligns the reference with the hardware, where
a row's lanes are adjacent registers rather than a stride away.

Worth recording because it is transferable: on this architecture the
batch dimension must be the fast-varying index. A fabric would not
suffer this, which is exactly why a CPU reference can mislead about
where the cost is.

## What these numbers do not support

- **They are not a prediction of FPGA behaviour.** `spm-stream-file`
  does not overlap IO; its two buffer slots refill synchronously, so
  `storage_time` and `compute_time` partition wall clock. `eta`
  measures a serial pipeline. On hardware fetch and compute overlap by
  construction, and `eta` would mean something different.
- **The store was warm.** 133 KB read repeatedly on a machine with
  64 GB of RAM is a page-cache read, not a disk read. The 8 GB/s
  figure is an upper bound on what any real parameter store will
  deliver, which makes the 196x gap a *lower* bound on the compute
  shortfall.
- **The engine is scalar reference code**, unoptimised on purpose. The
  196x is a property of this implementation, not a floor on what a CPU
  could do. It bounds the FPGA's target from one side only.
- **One matrix, one shape, one machine.** Nothing here says how the
  numbers move with matrix shape, group size, or a store that is
  genuinely slow.
