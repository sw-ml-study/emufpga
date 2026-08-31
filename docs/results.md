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

## Fabric model: where the pipeline stalls

Step 9 added the conceptual fabric model. Same 1024 x 512 `.spm` file,
8 weight lanes, batch width 8, batch 8, 512-byte FIFO, sweeping only
what the parameter store delivers per cycle:

```
emufpga sim -i bench.spm -b 8 -l 8 -w 8 -f 512 -F <fetch>
```

| fetch bytes/cycle | cycles | stall cycles | occupancy | cycles/weight |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 196608 | 131072 | 0.333 | 0.375 |
| 4 | 98304 | 32768 | 0.667 | 0.188 |
| 16 | 73728 | 8192 | 0.889 | 0.141 |
| 64 | 67584 | 2048 | 0.970 | 0.129 |
| 256 | 66048 | 512 | 0.992 | 0.126 |

The datapath's own cost is the floor: 65,536 cycles for 524,288
weights at 8 lanes, or 0.125 cycles per weight. Everything above that
is waiting on the store.

At 1 byte per cycle the pipeline spends a third of its cycles stalled
and occupancy is 0.333. Occupancy reaches 0.889 at 16 bytes per cycle
and 0.970 at 64; past that the returns are small, because the datapath
is already close to its own floor.

**This agrees in direction with the `eta` measurement above**, which is
the point of having both. The CPU bench found a serial pipeline whose
compute dominated by ~196x; the fabric model, given a datapath eight
weights wide, finds the crossing between store-bound and compute-bound
sitting somewhere between 4 and 16 bytes per cycle. Two different
models, the same qualitative answer: the arithmetic is the scarce
resource, not the bytes.

**Cycles are a unit, not a duration.** None of this converts to
seconds, and none of it says anything about whether such a datapath
fits any part. Both questions need measurements nobody has taken.

## TRM import (saga 2 step 2)

`yagizdevre/trm-maze-30x30` imported end to end. Verified manually
against the real checkpoint; the automated tests are hermetic and use
synthetic checkpoints, because weights never enter this repository.

```
scripts/extract-checkpoint model.pt extracted/
emufpga import -i extracted/ -o trm-maze.spm
```

| | |
| --- | ---: |
| Tensors | 15 |
| Parameters | 6,824,450 |
| `.spm` size | 27,324,980 bytes |
| Scale groups | 6,667 |
| Byte mismatches on round trip | **0** |

The byte accounting closes exactly:

```
  32               header
+ 480              directory, 15 descriptors x 32
+ 27,297,800       weights, 6,824,450 x 4
+ 26,668           inert scales, 6,667 groups x 4
= 27,324,980       observed file size
```

Round trip: every stream was read back group by group, rejoined, and
compared against the extractor's blobs. All 15 matched byte for byte,
and the payload was consumed exactly -- no trailing bytes, none
missing. Every stream carries the f32 profile and every scale is the
inert 1.0.

Scale overhead is 26,668 bytes on 27.3 MB, or **0.098%**, which is the
cost of the 1024-weight group size. The group buffer a reader must
hold is 4 KiB.

### What this does and does not establish

It establishes that a real trained checkpoint survives the path into
`.spm` unaltered. Nothing is quantized, reordered or rounded: the
extractor's blobs are already little-endian f32, which is exactly the
f32 encoding's wire format, so the importer is pure framing.

It establishes nothing about inference. No forward pass has been run
against this file -- that is the next step, and until it exists the
only claim here is that the bytes are intact.

## Consumption order (saga 2 step 3)

Step 2's importer laid the weights out alphabetically, because the
extractor uses `sorted()`. Reading TRM's own `trm.py` afterwards showed
that is the reverse of execution order:

| | per-layer order |
| --- | --- |
| Execution (from `trm.py`) | `qkv_proj, o_proj, gate_up_proj, down_proj` |
| Alphabetical (what shipped) | `down_proj, gate_up_proj, o_proj, qkv_proj` |

A forward sweep of that file would have to seek backward -- the one
thing this architecture forbids, and the exact opposite of
docs/research.txt's "arrange weights physically in exactly the order
the tensor engine consumes them".

`layouts/trm-maze-30x30.order` now specifies the order, in two
sections:

| section | streams | weights | share |
| --- | ---: | ---: | ---: |
| `[rotating]` | 8 | 6,815,744 | 99.87% |
| `[resident]` | 7 | 8,706 | 0.13% |

The rotating region must come first because `rewind` returns to the
start of the stream -- there is no seek to offer it an offset -- so
anything ahead of it would be re-read on every sweep for nothing.

The resident tensors are embeddings, init vectors and output heads.
Streaming 8,706 weights would be silly; docs/plan.md section 3 puts
small state in ordinary memory on purpose, and only the parameter bulk
is restricted.

### The property that is now tested

A forward pass is 15 `L_level` calls, each sweeping the 8-stream
rotating region once: 120 weight-matrix demands, 15 sweeps, 14
rewinds. The test walks that demand sequence and asserts the stream
index never decreases except at a rewind, and that every rewind
happens only after the sweep finished. It runs against the real
tracked order file with no checkpoint download -- the weights would be
needed to test values, not layout.

A second test pins the mistake itself: sorting the rotating names
alphabetically puts `o_proj` before `qkv_proj`, so a sweep seeks back.
It exists so the regression cannot quietly return.

## TRM forward pass (saga 2 step 4)

Rung 1 of the model ladder. The recursion runs over streamed weights,
rewinding the rotating region between every `L_level` call.

### What is streamed and what is not

| | where | size |
| --- | --- | ---: |
| The four projections per layer | streamed | 99.87% of weights |
| `z_H`, `z_L`, residuals, norms, `RoPE`, softmax, `SwiGLU` | resident | kilobytes |

docs/plan.md section 3 allows the second column in ordinary memory on
purpose. The asymmetry between the columns is the reason the
architecture works: a softmax needs a whole row before it can
normalize, but it touches kilobytes, while the weights are megabytes
and need nothing but to arrive in order.

### Measured

| | |
| --- | ---: |
| `L_level` calls per forward | 15 |
| Rewinds per forward | 14 |
| Rotating region re-read | 15x per forward |
| Streams swept per call | 8 |

### `Ps` under-reports recursion by 15x

Scan productivity as `spm-stream-metrics` defines it counts
applications per weight read, and a scan-level view sees batch reuse
only. For a recursive model that is badly wrong: TRM re-reads its
entire rotating region 15 times per forward, so weights are reused
across **depth** as well as across batch.

`Forward::scan_productivity` counts both, giving `positions * calls`.
At 8 batch positions that is 120 rather than 8 -- and at
`halt_max_steps` 16 a full puzzle reaches 240 sweeps, so the gap grows
further. A dense transformer touches each weight once per token; TRM
touches its whole weight set fifteen times per forward. The
architecture's central problem is amortizing a scan, and this model
solves it for us.

### Bit-exact, and what that does not mean

The streamed matmul agrees with a resident one **bit for bit** on
TRM's real shapes (1536x512, 512x512, 3072x512, 512x1536), and
batching changes no answer: each position gets exactly what it would
have got alone. Both use `mul_add`, since a fused multiply-add rounds
differently from a separate multiply and add.

That establishes **mechanism**, not correctness of the model. Nothing
here shows TRM solves mazes. Verifying that needs the published
implementation, and torch is not installed -- it is the next step, and
until then the only claim is that streaming changes nothing about the
arithmetic.

Two assumptions are recorded rather than hidden: the `RoPE` base is
taken as 10000 because the published value is not in the config we
have, and attention is unmasked because TRM sees a whole maze at once
rather than generating left to right. Both get confirmed against the
reference.

## Verified against the reference (saga 2 step 4)

The block now agrees with the published implementation, and getting
there found two bugs that every test written so far had missed.

Method: a torch script builds one TRM block from `models/layers.py`'s
formulas, runs it on fixed random weights, and dumps every
intermediate. The Rust side imports the same weights and reproduces
each stage. Manual, because it needs torch and 13 MB of weights, and
weights never enter this repository.

| stage | max abs error | relative | cosine |
| --- | ---: | ---: | ---: |
| qkv projection | **0.0** | 0.0 | 1.0000000000 |
| RoPE + attention | 5.96e-8 | 2.1e-7 | 1.0000000000 |
| o_proj + residual + norm | 9.54e-7 | 2.7e-7 | 1.0000000000 |
| MLP + residual + norm | 9.54e-7 | 2.7e-7 | 1.0000000000 |

Tolerance, not bit-exactness, and the distinction is deliberate.
Bit-exactness is claimed only between this project's own streamed and
resident paths, where the summation order is identical by
construction. Against torch it cannot hold: its GEMM and fused
attention accumulate differently. What agreement at 1e-7 shows is that
the **formulas** match.

### Bug 1: the intermediate width

`SwiGLU`'s intermediate is `round(expansion * hidden * 2/3)` aligned up
to a multiple of 256 -- **1536** for TRM, not `hidden * expansion` =
2048 as this crate assumed. The checkpoint proves it: `gate_up_proj`
is `(3072, 512)` and 3072 is 2 x 1536.

The tests missed it because they generated their shapes from the same
wrong formula they were checking. Self consistent, and self
confirming. The fix ships with a test written against the published
numbers instead.

### Bug 2: rms_norm normalized the wrong axis

The reference computes `hidden_states.pow(2).mean(-1, keepdim=True)`
-- a mean over the last axis, so **every position is normalized by its
own RMS**. This crate normalized the whole state as a single vector.

That is wrong in a way that is easy to miss. With a handful of
positions the scales are similar, the output stays finite and
plausible, and nothing about it looks broken. It cost a cosine of
0.9993 while the stages either side were exact -- which is exactly why
the bisection existed rather than a single end-to-end number.

### Bug 3: every imported matrix was transposed

The worst of the three, and the one the existing tests were least able
to see. PyTorch stores row-major, so `W[r][c]` sits at `r*cols + c`.
`.spm` stream order is column-major: index `k` holds
`W[k % rows][k / rows]`. `scripts/extract-checkpoint` dumped raw
storage bytes, so every matrix in the imported file was its own
transpose.

The round-trip test passed throughout, because **round-tripping bytes
says nothing about how they are interpreted**. Only a numerical
comparison against a reference could show it.

The conversion now happens in the extractor, which is the boundary
between a framework's conventions and the format's. Verified against
torch ground truth: `got[k] == W[k % rows][k / rows]` for the real
checkpoint's `qkv_proj`, and the old row-major layout no longer
matches.

That keeps the Rust importer pure framing -- it still never touches a
value. The earlier claim that it was "pure framing" was true of the
code and false of the pipeline, because the bytes it framed did not
yet mean what the format said they meant.

## HRM, rung 2 (saga 2 step 5)

`zbloss/HRM-sudoku-extreme`, 27,276,802 parameters. Imported, laid out
in consumption order, and driven through its two-module recursion.

| | |
| --- | ---: |
| Tensors | 39 |
| Parameters | 27,276,802 |
| Rotating streams | 32 (4 matrices x 4 layers x 2 modules) |
| Rotating weights | 27,262,976 (99.95%) |
| Resident weights | 13,826 (0.05%) |
| `.spm` size | 109,215,052 bytes |

### The question this rung opened, and the answer

HRM has **two** modules where TRM has one shared block:

```text
for h in range(H_cycles):        # 2
  for l in range(L_cycles):      # 2
    z_L = low_level(z_L, z_H + input)
  z_H = high_level(z_H, z_L)
```

The risk was that two modules force two rotating regions -- which
`rewind` alone cannot serve, because it returns to the start of the
stream and there is no seek to offer it an offset.

**They do not.** With `[low][high]` contiguous and low first, a plain
rewind-to-zero serves the whole recursion:

```text
sweep low            cursor -> first high stream
rewind, sweep low    cursor -> first high stream
sweep high           cursor -> end
rewind               cursor -> first low stream
```

The last low sweep of an outer cycle leaves the cursor exactly where
the high module begins, so the high sweep continues forward with no
seek. One rewind before every low sweep except the first, and never
before a high sweep: `h_cycles * l_cycles - 1` in total.

That works **only** because low precedes high. The checkpoint's own
ordering is alphabetical, which puts `high_level_module` first and
would require a backwards seek -- the same trap TRM had, in a second
place, and the postmortem predicted it.

### What is verified and what is inferred

The block is verified. HRM's source describes it as self-attention,
RMS norm, fully connected, RMS norm with post-norm residuals -- the
same block TRM uses, which is unsurprising since TRM was derived from
HRM -- and its config states the same `hidden_size`, `num_heads`,
`expansion`, `intermediate_size`, `rope_theta` and `rms_norm_eps`. TRM's
block is numerically verified against its published implementation at
cosine 1.0, and `spm_trm::Layer` is reused unchanged.

### Verified against the official implementation

The first version of this section said the recursion could not be
checked, because `transformers` does not recognise `model_type: hrm`
and the checkpoint ships no modeling code. That was giving up too
early: `sapientinc/HRM` **is** the modeling code, and the ported
checkpoint's tensors differ from it only in name.

Cloned it and ran its own `ReasoningModule` on the real weights. The
one obstacle is `flash_attn`, which is CUDA-only; it is a performance
kernel rather than a different algorithm, so a stand-in built on
torch's `scaled_dot_product_attention` computes the same function. The
remapped tensors load with **zero missing and zero unexpected keys**,
which is itself evidence the shapes and the mapping are right.

One low-level module sweep, four layers, real weights, 8 positions:

| | |
| --- | ---: |
| max abs error | 1.43e-6 |
| relative | 4.16e-7 |
| cosine | **1.000000000000** |

### The bug that found

Writing that comparison exposed an omission the earlier tests could
not see: `forward` never performed HRM's **input injection**.
`ReasoningModule.forward` adds an injection to the hidden state before
its layers run, and the recursion supplies a different one each time --
`z_high + input` for the low module, `z_low` for the high one. This
crate was computing `low(z_L)` where HRM computes
`low(z_L, z_H + input)`.

The result ran, stayed finite, produced plausible numbers, and was not
HRM. The tests checked sweep counts, rewind counts and finiteness, all
of which passed. It is the same failure mode docs/postmortem-1.md
records for TRM's layout bugs, repeated one rung later for semantics
rather than layout -- which is a reasonable argument that the
postmortem's rules are not yet strong enough, since they were written
down and the bug still landed.

## Streamed against conventional resident (saga 2 step 6)

The first head-to-head. Plan item 5, deferred past the HRM rung: two
rungs of the ladder had been built and nothing had yet compared the
streamed path against the conventional one it claims to replace.

### What was measured

TRM's full forward pass -- 15 `L_level` calls over 2 layers, on the
real 6,824,450-parameter checkpoint -- in three configurations:

| | weights live | reached by |
| --- | --- | --- |
| resident | all in RAM | subscript |
| streamed (memory) | one group at a time | arrival |
| streamed (file) | one group at a time | arrival, off disk |

Three rather than two on purpose. A file-versus-resident number alone
conflates the cost of streaming with the cost of storage; the
memory-backed stream keeps the streaming discipline and removes the
IO, so the two costs can be told apart.

`components/tensor/crates/spm-trm-resident` is the conventional path.
It is a separate crate rather than a generic parameter on `spm-trm`,
because abstracting over "weight source" would require a trait that
admits random access, and the streamed path's guarantee is that random
access is **not expressible in its types**. The contrast is the point;
hiding it behind a generic would erase it.

### Correctness first

All three configurations agree **bit for bit** on the real checkpoint,
at every batch size measured: 0 mismatched floats out of 8 x 512. Both
paths apply weights in the same order and both use `mul_add`, so there
is no rounding difference left for a tolerance to absorb.

This is a precondition, not a result. A performance comparison between
two implementations that compute different answers is not a
comparison, and `spm-trm-resident/tests/agreement.rs` asserts it
hermetically on synthetic weights so it cannot quietly rot.

### The numbers

M1 Max, release build, best of 5. Times are milliseconds per forward.

| batch | resident | streamed (mem) | streamed (file) | store MB/s demanded |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 102 | 116 (+13.9%) | 143 (+40.7%) | 2854 |
| 4 | 308 | 341 (+10.7%) | 364 (+18.3%) | 1123 |
| 8 | 590 | 610 (+3.5%) | 642 (+8.9%) | 637 |
| 16 | 1136 | 1163 (+2.4%) | 1194 (+5.1%) | 342 |
| 32 | 2402 | 2438 (+1.5%) | 2461 (+2.5%) | 166 |

Parameter bytes held in random-access memory, unchanged across every
row:

| | bytes |
| --- | ---: |
| resident | 27,297,800 |
| streamed, either backing | 4,096 |
| ratio | 1.50e-4, or 6,664x |

Repeating the whole sweep moves these by about 1%: two batch-32 runs
gave 2402/2438/2461 and 2400/2411/2443 ms. **The +1.5% memory-stream
overhead at batch 32 is therefore at the noise floor**, and should be
read as "no longer measurable on this machine" rather than as a
precise figure. The batch-1 and batch-4 overheads are many times the
spread and are real.

### Finding 1: streaming costs little, and less the more you batch

The overhead falls monotonically with batch, from 41% at batch 1 to
2.5% at batch 32. That is the amortization the architecture is built
on, measured end to end for the first time: a weight is fetched once
and applied `positions` times, so the fixed cost of getting it there
is divided by the batch.

Splitting file from memory attributes it. At batch 32 the streaming
mechanism itself costs 1.5% and the file adds a further 1.0%. At batch
1 the mechanism costs 13.9% and the file adds 26.8% -- the same
per-group overhead, spread over a thirty-second of the work.

### Finding 2: the demanded store bandwidth falls with batch

The rotating region is 27,262,976 bytes and is re-read once per
`L_level` call, so a forward pass pulls 408,944,640 bytes regardless
of batch. The **rate** that traffic must arrive at is what changes:
2.9 GB/s at batch 1, 166 MB/s at batch 32.

This is the number the thesis turns on. At batch 1 the store must be
RAM or NVMe. At batch 32 a single SAS drive would keep the engine fed
with room to spare, which is the substitution the project exists to
test: cheap sequential capacity in place of expensive fast memory.

**The honest caveat is large and cuts the other way.** The demanded
rate is low partly because the compute is slow. This is scalar
reference code; a 10x faster engine at batch 32 would demand 1.7 GB/s
and put the disk back on the critical path. The rate is a property of
the ratio between this engine and this store, not a property of the
architecture. It bounds nothing on its own.

### Finding 3: the memory win is real, and not yet load-bearing

6,664x fewer resident parameter bytes is the largest number in this
document, and at this rung it buys nothing. 27 MB fits in any device
made this century. The streamed path's residency is also **O(1) in
model size** -- the buffer is sized by the widest group, not by the
model, so a 300 GB model streams through the same 4 KiB.

Where it starts to matter, at f32:

| VRAM | parameters held resident | TRM as a fraction |
| --- | ---: | ---: |
| 8 GB | 2.00 G | 0.34% |
| 12 GB | 3.00 G | 0.23% |
| 24 GB | 6.00 G | 0.11% |

So the resident path stops fitting an 8 GB card somewhere around 2
billion f32 parameters, roughly 300x up the ladder from here. That is
the rung at which this comparison stops being an academic 2.5% and
becomes the difference between running and not running -- and it is
why the ladder goes on rather than stopping at a favourable number.

### What these numbers do not support

- **The resident path is not an optimised GEMM.** It is the same
  scalar loop with the weights already in RAM, chosen deliberately so
  the arithmetic is held fixed and the difference is attributable to
  the mechanism. A real inference engine using BLAS would beat both
  columns by a wide margin. Nothing here says the Serial Parameter
  Machine is faster than conventional inference; it says streaming
  costs 2.5% against the same arithmetic.
- **The file was page-cached.** 27 MB re-read fifteen times on a
  64 GB machine never touched a disk. The file column measures syscall
  and copy overhead, not storage. A cold store is exactly what Finding
  2's demanded-rate column is for, and that column is a requirement,
  not a measurement of any real device.
- **The memory axis measures parameters, not RSS.** Activations,
  buffers and the allocator are excluded on both sides. The claim is
  about where the weights are, which is the claim the architecture
  makes.
- **Streaming from a page-cached file does not reduce system-wide
  memory.** The kernel still holds those pages. The 4 KiB figure is
  process-resident parameter bytes; the win becomes a real system-wide
  win only when the store is a device you are not caching, which is
  the configuration this machine cannot produce.
- **One machine, one model, one precision.** Every standing caveat
  from the batch-amortization measurement still applies: no IO
  overlap, scalar engine, warm cache.

## BDH, rung 3 (saga 2 step 7)

`pathwaycom/bdh`, the Dragon Hatchling (arXiv 2509.26507). The first
rung whose result **contradicts** an assumption the plan rests on.

### What BDH is, structurally

Reference defaults: `n_layer` 6, `D` 256, `nh` 4,
`mlp_internal_dim_multiplier` 128, `vocab` 256, so the sparse latent
is `N = 128 * 256 / 4 = 8192` -- thirty-two times the model width.

| tensor | shape | weights |
| --- | --- | ---: |
| `encoder` | (nh, D, N) | 8,388,608 |
| `encoder_v` | (nh, D, N) | 8,388,608 |
| `decoder` | (nh*N, D) | 8,388,608 |
| `lm_head` | (D, vocab) | 65,536 |
| `embed` | (vocab, D) | 65,536 |

**The loop body carries no layer index.** BDH applies one parameter
set `n_layer` times, so it is a rotating parameter store by
construction -- the property that made TRM a good first rung, arriving
here for free rather than by design. 9 rotating streams, 5 rewinds per
forward, 604 MB of traffic for a 101 MB model.

`lm_head` is read exactly once, after the last level, so it sits
**after** the rotating region: the final `decoder` sweep leaves the
cursor exactly where `lm_head` begins, and the logits are produced by
reading on rather than by seeking back. Same structure as HRM's
`[low][high]`, and verified rather than assumed
(`lm_head_sits_after_the_rotating_region`).

Attention carries no learned parameters at all. Its RoPE frequencies
are computed, and it rotates over the sparse latent `N`, not over a
head dimension of the model width -- a different operator wearing a
familiar name.

### Verified against the reference

`pathwaycom/bdh` ships architecture and training code but no trained
checkpoint, so the comparison uses seeded random weights exported from
torch. That still checks every formula exactly, which is what caught
the bugs on TRM and HRM.

| stage | cosine | relative |
| --- | ---: | ---: |
| `x_embedded` | 1.000000000000 | 1.86e-7 |
| `x_after_0` | 1.000000000000 | 1.32e-6 |
| `x_after_5` | 0.999999999998 | 2.39e-6 |
| `logits` | 0.999999999998 | 2.52e-6 |

Residual f32 accumulation noise, nothing more.

**This is the first rung that worked on the first run.** No bisection
was needed because none of the stages disagreed. The difference from
TRM and HRM is not luck: the reference forward pass was read *before*
any code was written, which is the actionable change postmortem 1
prescribed and which the HRM rung followed only partly. Three details
would each have been a silent bug if guessed -- `get_freqs` quantizes
the index in pairs so a rotation pair shares a frequency;
`.tril(diagonal=-1)` is **strictly** lower triangular, so a position
never attends to itself; and `nn.LayerNorm` subtracts the mean, where
TRM and HRM use RMS norm and do not.

### The transpose is not universal

`scripts/extract-checkpoint` transposes every 2-D tensor, because a
torch `Linear` stores `(out, in)` while `.spm` wants column-major
stream order. **BDH stores every parameter as `(in, out)` instead**,
so its raw row-major bytes already *are* stream order and the declared
shape is simply reversed. Running the generic extractor over BDH would
have reintroduced postmortem defect 8 in mirror image.

The transform is a property of the source framework's storage
convention for a given tensor, not a universal rule -- which is the
"bytes versus meaning" lesson recurring in a new place. BDH gets its
own exporter for this reason.

### The finding: the sparse latent overtakes the weights

docs/plan.md section 3 keeps activations in ordinary memory on the
grounds that they are kilobytes while the weights are megabytes. For
BDH that is true only at short sequences, and it does not merely
weaken with length -- it inverts.

Resident activation bytes against a fixed 100,925,440-byte weight set:

| positions | activations | as % of weights |
| ---: | ---: | ---: |
| 16 | 3.3 MB | 3.2% |
| 64 | 13.0 MB | 12.9% |
| 128 | 26.0 MB | 25.7% |
| 256 | 51.9 MB | 51.5% |
| 512 | 103.8 MB | **102.9%** |

At 512 positions the engine holds more activation than model. The
crossover is near 498 positions, and it is inevitable rather than
incidental: the latent is `positions * heads * latent` and grows
linearly, while the weight set does not grow at all. `budget()`
reports both, and `the_sparse_latent_overtakes_the_whole_weight_set`
asserts the inversion so it cannot be quietly lost.

This engine already spends half of what the reference would. Folding
the second `relu` into an in-place gate lets one buffer serve as both
`x_sparse` and `xy_sparse`, where the reference holds both at once.
The inversion above is *after* that saving.

### What it means for the FPGA target

On the Tang Nano 9K the scarce resource is BRAM, and it is measured in
kilobytes. An architecture whose working set is tens of megabytes and
grows with sequence length does not fit the "static parameters
outside, dynamic state inside" partition that docs/research.txt
proposed for BDH -- not because the parameters are too big, which is
the problem the Serial Parameter Machine solves, but because the
*state* is. Streaming the weights perfectly would not help.

That is a real negative result for BDH-on-FPGA at these
hyperparameters, and it is specific: `mlp_internal_dim_multiplier` is
what drives it. At 128 the latent is 32x the model width. A smaller
multiplier would move the crossover, and nothing measured here says
what BDH's quality does when it moves.

### What these numbers do not support

- **Random weights, not a trained model.** Every formula is verified,
  and nothing about BDH's behaviour or quality is. `pathwaycom/bdh`
  ships no checkpoint; if one appears, the comparison should be rerun
  against it.
- **One configuration.** The residency inversion is a property of
  `mlp_internal_dim_multiplier = 128` and the reference `D` and `nh`.
  It is arithmetic, so it will move predictably, but no other setting
  was measured.
- **No timing is reported.** This rung establishes structure and
  residency. The streamed-versus-resident comparison that step 6 did
  for TRM has not been done for BDH.
- **`embed` is excluded from the streamed path deliberately.** A
  lookup table gathered by token id cannot be swept to serve one
  token. At 65,536 weights, 0.26% of the model, it stays resident --
  plan section 3 working as intended rather than an exception to it.
- **The attention cost is not addressed here.** Scores are
  `positions^2` per head and the engine computes them without
  materialising the matrix, but nothing measures whether that is the
  right trade at long sequences.
