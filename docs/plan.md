# emufpga -- implementation plan

Serial Parameter Machine (SPM) research vehicle: a conceptual model of
something FPGA-like, driving a streaming low-bit tensor engine, aimed
eventually at the Gowin parts on Sipeed Tang Nano boards.

Source: `docs/research.txt`.

## 1. What this repository is

`docs/research.txt` proposes inverting the accelerator memory model:

- Conventional inference treats gigantic immutable weight matrices as
  though they need general-purpose random-access memory, even though
  the dominant operation traverses them in a fully predictable order.
- The SPM proposition: put the weights in cheap sequential storage,
  move compute to the weight stream, and keep only activations,
  accumulators, scales and recurrent state in fast memory.

The research names three fronts: a Linux/CPU simulation front, an FPGA
front, and an RP2350/PIO front. This repository is the **FPGA front's
software half**, built so its outputs are also the golden model for the
other two.

**emufpga is** a conceptual, cycle-approximate model of an SPM
streaming tensor engine. It answers two questions for a proposed engine
configuration:

1. Is it correct? (bit-exact against a CPU reference over identical
   golden vectors)
2. Where does the pipeline stall -- does the datapath starve waiting on
   the parameter stream, or does the stream back up waiting on the
   datapath?

Its knobs are abstract: lanes, FIFO depth, fetch rate. **Cycles are a
unit, not a duration.** Nothing here converts to seconds, because no
fabric clock has been measured and inventing one is how a conceptual
model quietly becomes a fidelity claim.

**emufpga is not** a gate-level or bitstream-accurate Gowin simulator,
it does not emit HDL, and it does **not** predict whether a design fits
a given part. No LUT4 budgets, no utilization percentages, no
place-and-route predictions. RTL is written by hand later and validated
against the golden vectors this repository produces.

That boundary is deliberate and was tightened once, in step 8. The
original plan had a resource-budget and fit report checked against real
device profiles; it was withdrawn before being written, because
high-fidelity fit modelling is the opposite of conceptual exploration
and would also have been unfalsifiable -- the two figures a fit model
most needs, bulk memory bandwidth and fabric fmax, are `Unknown` for
every board (section 6). Something FPGA-like that can be refined later
beats something device-shaped that cannot be checked.

## 2. Decisions taken

| Question | Decision |
| --- | --- |
| Emulator depth | Conceptual cycle-approximate model with abstract knobs. No HDL emission, no fit prediction. Narrowed from "cycle model + resource budget" at step 8. |
| Primary device | Tang Nano 9K (Gowin GW1NR-9). Other boards are added as profile data, not as new code. |
| External toolchain | Open source: Yosys, nextpnr-himbaechel, Project Apicula, openFPGALoader. Chosen so a real place-and-route can be scripted and diffed against the emulator's prediction. |
| First workload | Synthetic ternary GEMV. Research "Phase 0: don't run a model". Real models arrive in saga 2+. |

Rationale for the last one: the walking skeleton must prove the format,
the seek-free stream, ternary accumulation, batch amortization and the
fit report as one vertical slice, with exact golden vectors isolating
each layer. Importing a real model first couples format bugs, engine
bugs and model-import bugs together with nothing to bisect against.

## 3. The abstract machine

One interface, shared by every front, stated as a restriction rather
than a capability:

```
WeightStream
    next_block(&mut self, dst: &mut [u8]) -> Result<usize>
    rewind(&mut self)          -- between operations only
    -- NO seek. NO addressing. NO random access.

TensorSink
    begin(&mut self, op: &Op, activations: &[Activation])
    consume(&mut self, weights: &[u8])
    finish(&mut self, out: &mut [Accum])
```

**The experimental rule:** the tensor engine may never seek backward or
randomly into the parameter stream while performing an operation.
Activations, accumulators, scales, routing state and KV may use ordinary
random-access memory. The rule is enforced in the type system -- there
is no seek method to call -- so the architecture cannot quietly erode
back into a memory controller as the code grows.

**Ternary is the one change that moves both terms at once**, and it is
the open research direction with the strongest case. It is 16x smaller
AND needs no multiplier, so a given store feeds more lanes and more
lanes fit in the fabric. See docs/research-ternary-fpga.md, which also
works out where a PC/FPGA boundary would have to fall -- weights must
never cross the host link, activations may -- and finds that the
partition is the same asymmetry saga 2 step 11 measured. Nothing there
is measured yet; the ternary profile has never run against a real
model.

**Streaming is a demonstration rather than a win whenever the working
set fits in memory anyway.** Rung 4 made this unavoidable. Arithmetic
intensity is `batch / 4` MACs per weight-byte for any f32 model read
once, recursion or not -- so recursion does not buy reuse, it buys a
small working set. TRM's 27 MB rotating region fits in any machine's
RAM, and caching it beats streaming it. The first three rungs were all
in that regime, and the architecture looked good there for a reason
that does not generalise.

The case for a serial parameter store is a model too big to hold, read
once, in order. SmolLM2-135M is the first rung that is actually that,
and it is the one to reason from. See docs/results.md, saga 2 step 8.

**The permission granted to activations is not free, and BDH is where
that showed.** Letting activations use ordinary memory is justified by
an asymmetry -- weights are megabytes, activations are kilobytes -- and
that asymmetry is a property of the architecture being run, not a law.
BDH's sparse latent is `positions * heads * latent` floats and grows
linearly with sequence length while its weight set does not grow at
all. At 512 positions it holds more activation than model (103.8 MB
against 100.9 MB); see docs/results.md, saga 2 step 7.

The rule above still stands, because it is about the weight stream and
nothing here touches it. What does not stand is the assumption that
obeying it is sufficient. **For any new rung, the resident working set
is a number to measure before the rung is planned, not a detail to
discover afterwards** -- and for a target whose fast memory is BRAM
measured in kilobytes, it can disqualify an architecture whose weights
would have streamed perfectly.

The `.spm` container is a **physical execution layout**, not another
model interchange format. Weights are stored in exactly the order the
engine consumes them; opening a stream and reading to the end IS the
matrix operation.

## 4. Metrics

Every benchmark reports these. GB/s alone does not express what the
architecture is trying to improve.

| Metric | Definition | Why |
| --- | --- | --- |
| Raw sequential bandwidth | bytes/s off the backing store | physical limit |
| Decoded weights/s | after unpacking | representation overhead |
| Useful weights/s | into accumulators | engine rate |
| `eta` | engine consumption bandwidth / storage bandwidth | can compute keep up with storage? If not, the hardware design has its target. |
| `Ps` (scan productivity) | useful parameter applications / parameter values read | the entire economic argument. Batch 1 gives Ps ~= 1; batch 16 gives ~16; MoE batching more. |
| `Rp` (parameter residency) | parameter bytes resident in RAM / total parameter bytes | conventional inference is ~1. Goal: Rp -> 0 while activation RAM stays nonzero. |
| Numerical error | max abs, mean, cosine vs f32 reference | correctness, and the regression suite for later RTL |

For the emulator specifically, add: LUT4 / DFF / BSRAM-18Kb / DSP18x18 /
IO utilization per device profile, cycles per operation, and predicted
wall-clock at the profile's fmax.

## 5. Repository architecture

Modelled on `../sw-mlpl`. **No root `Cargo.toml`.** Each top-level
directory under `components/` is its own cargo workspace. This is not
cosmetic: it is how the implementation spreads horizontally enough to
stay inside the complexity gates.

```
emufpga/
  components/          each dir = one cargo workspace
    format/
    stream/
    tensor/
    device/
    fabric/
    engine/
    report/
    cli/
  scripts/             all build/test/gate entry points
  docs/
  models/              synthetic fixtures, later trm/
  .cargo/config.toml   shared target/ dir
  justfile             thin delegation to scripts/
```

### Complexity gates

Target gates, stricter than the `sw-checklist` FAIL line, per
`../sw-mlpl` CLAUDE.md:

- 25 LOC per function
- 4 functions per module
- 4 modules per crate (`lib.rs` counts, so: a facade plus three)
- 350 LOC per file
- No automated gate on crates per component -- let the module ceiling
  decide how many crates a component needs

`lib.rs` and `mod.rs` are facades only -- no executable logic. Behavior
lives in named files following the convention `parse.rs` (input ->
typed), `validate.rs` (typed -> result), `plan.rs` (config -> plan),
`run.rs` (effects), `render.rs` (data -> string), `error.rs`,
`model.rs` (data types), `test_support.rs`, `fixtures.rs`.

Split by responsibility, never mechanically, and never with whitespace
tricks, `rustfmt::skip` or `#[allow(...)]`. Do not add new logic to an
already over-limit function or module: extract first, then add.

Edition 2024. `[workspace.lints.clippy] pedantic = "warn"`.
`cargo clippy --all-targets --all-features -- -D warnings` must be
clean.

### Component and crate map

The full target shape. Saga 1 builds a subset (marked *).

**`components/format/`** -- the `.spm` container (built, saga 1 step 2)

| Crate | Responsibility |
| --- | --- |
| `spm-bytes` * | little-endian integer primitives |
| `spm-header` * | magic, version, endianness, stream count |
| `spm-codec` * | ternary bit packing; Q4 arrives in saga 3 |
| `spm-layout` * | op descriptors, consumption-order tiling, scale-group arithmetic |
| `spm-walk` * | forward-only cursor over the (stream, group) sequence |
| `spm-file` * | reader/writer composing the above |

Six crates rather than the four first sketched: the four-module
ceiling pushed the byte primitives and the group cursor out into their
own crates. The cursor in particular earns its place -- reader, writer
and the step 003 `WeightStream` all walk the same group sequence, and a
disagreement between them would surface as plausible wrong numbers
rather than an error.

**`components/stream/`** -- sequential parameter access (built, saga 1 step 3)

| Crate | Responsibility |
| --- | --- |
| `spm-stream` * | `WeightStream` trait and error; seek-free by construction |
| `spm-stream-mem` * | in-memory impl; the reference other backends must match |
| `spm-stream-file` * | file impl over a two-slot buffer pair |
| `spm-stream-groups` * | scale groups pulled off any `WeightStream` |
| `spm-stream-metrics` * | bandwidth, `eta`, `Ps`, `Rp` |

`spm-stream-groups` was not in the original sketch. `spm-file`'s
reader walks a file already in RAM; the streaming path needs the same
walk over bytes that are still arriving, which is the case the whole
premise depends on. Its `resident_parameter_bytes` is what makes `Rp`
a measurement rather than an assertion.

`eta` is defined as `storage_time / compute_time`: both bandwidths are
the same byte count over a different time, so the ratio reduces. Above
1 the scan is storage-bound, which is the regime the architecture
wants; below 1 the tensor engine has a concrete speedup target.

**`components/tensor/`** -- CPU golden reference (built, saga 1 step 4)

| Crate | Responsibility |
| --- | --- |
| `spm-accum` * | accumulator banks with a batch dimension |
| `spm-activations` * | resident activations, and the only multiply |
| `spm-numeric` * | naive f64 reference matmul, error metrics |
| `spm-gemv-ref` * | ternary GEMV over a `WeightStream`; add / sub / skip, no multiplier |
| `spm-vectors` * | reproducible golden case generation |
| `spm-vectors-text` * | golden case text format |

**Accumulator width is an open question, not a settled one.** The
`Ternary2F32I32` profile name says `i32`; building the reference showed
that name had settled something not actually worked out. Scale groups
run along the stream and the stream is column-major, so a group's
scale can vary with both output row and input column. Two designs keep
the inner loop multiplier-free: pre-scale the activation (needs `f32`
accumulation, exact) or pre-scale into fixed point (allows `i32`,
cheaper in LUTs, introduces rounding). The reference takes the exact
one, because an oracle carrying its own quantization error cannot
adjudicate anyone else's. Saga 2 decides, with the fabric to measure
against. No format change was needed: the profile discriminant is a
wire value and nothing on disk depends on accumulator width.

**`components/device/`** -- Gowin device profiles (built, saga 1 step 7)

| Crate | Responsibility |
| --- | --- |
| `gowin-profile` * | profile data types, the Tang Nano board table |

**Reference data with no consumer, by design.** `gowin-budget` and
`gowin-timing` were planned here and are not coming: they existed to
serve the withdrawn fit report. Nothing in the tree imports
`gowin-profile`, which is what keeps the fabric model from drifting
back toward device prediction.

**`components/fabric/`** -- the conceptual fabric model (step 9)

| Crate | Responsibility |
| --- | --- |
| `fabric-model` | abstract config, execution, cycle accounting |
| `fabric-report` | rendering a run |

Knobs are abstract -- `weight_lanes`, `batch_width`, `fifo_bytes`,
`fetch_bytes_per_cycle`, `fetch_latency_cycles` -- and none of them is
tied to a part. Blocks do **not** declare resource costs; that was the
withdrawn fit model's idea and it is not coming back at this stage.

**`components/cli/`** -- binaries (`pack` built, saga 1 step 5)

| Crate | Responsibility |
| --- | --- |
| `spm-quantize` * | dense f32 matrix to ternary `.spm`, dependency-free |
| `spm-bench` * | the batch-amortization sweep |
| `spm-bench-report` * | rendering a sweep as markdown |
| `emufpga-cli` * | argument parsing and subcommand dispatch (clap) |
| `emufpga` * | thin binary: parse, dispatch, print, exit code |

Subcommands: `pack` and `bench` (steps 5-6), `fit` (step 8), then
`sim` and `verify` in saga 2.

**Step 6 measured the make-or-break experiment; see docs/results.md.**
The crossover it set out to find does not exist in range: the CPU
reference is compute-bound at every batch size, roughly **196x too
slow to saturate even a page-cached file read**. That ratio is the
concrete target step 008's fit model must be held to.

`clap` is the repository's only external dependency, and
`sw-checklist` has a check that looks for it specifically.
`spm-quantize` stays dependency-free so saga 4's model import can reuse
the quantization rule without pulling a CLI parser in.

**Quantization is per-group absmean** (the BitNet b1.58 rule):
`scale = mean(|w|)` over the group, `t = clamp(round(w / scale), -1, 1)`,
ties away from zero. An all-zero group is written as `scale = 1.0` with
every weight `Zero`, keeping a zero scale out of the wire format. The
transform is lossy by design and the rule is pinned by tests rather
than left implicit.

## 6. Device profiles (reference data, no consumer)

`components/device/` is **cited reference data with nothing built on
it.** It cost one step, the figures are sourced and tested, and saga 6
will want them when real place-and-route enters the picture. Until
then nothing imports `gowin-profile`, deliberately: a fabric model that
could reach for LUT4 counts would drift back toward the fit modelling
this project has decided not to do yet.

Target boards, all owned. Figures are **sourced**, not remembered --
the table this section used to carry was written from memory and is
gone. Every value below is cited in
`components/device/crates/gowin-profile/src/boards.rs` with the
document and retrieval date, and anything that could not be sourced is
`Unknown` in the type system rather than a plausible number.

| Board | Part | LUT4 | FF | B-SRAM | DSP 18x18 | PLL | Bulk memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tang Nano | GW1N-1 | 1,152 | 864 | 72 Kb (4) | unknown | 1 | PSRAM 64 Mb |
| Tang Nano 4K | GW1NSR-LV4CQN48PC6/I5 | 4,608 | 3,456 | 180 Kb | unknown | 2 | PSRAM in package, size unstated |
| **Tang Nano 9K** | **GW1NR-LV9QN88PC6/I5** | **8,640** | **6,480** | **468 Kb (26)** | **20** | **2** | **PSRAM 64 Mb** |
| Tang Nano 20K | GW2AR-LV18QN88C8/I7 | 20,736 | 15,552 | 828 Kb (46) | 48 | 2 | SDR SDRAM 64 Mb, 32-bit |
| Tang Nano 25K | GW5A-25 | 23,040 | unknown | 1,008 Kb | 28 | 6 | unconfirmed |

Sources: the Sipeed wiki page for each board, retrieved 2026-08-30.
Gowin's DS117-3.2.5E was retrieved as the family authority for the
GW1NR parts, but its resource tables are CID-encoded and did not
extract to text on this machine, so the figures come from the vendor
pages that restate them.

### The two gaps that matter

**Bulk memory bandwidth is unknown for every board.** This is the most
consequential gap in the project. The architecture's premise is
trading random access for cheap sequential bandwidth, and step 6
measured the CPU reference as roughly 196x too slow to saturate even a
page-cached read (docs/results.md). Which side of that ratio a board
lands on is decided by what its PSRAM or SDRAM sustains -- a figure
board documentation does not state, because it depends on the memory
controller as much as on the part. Measuring it is hardware work.

**Achievable fabric fmax is unknown for every board.** Datasheets give
per-primitive timing, not a fabric-wide number. The honest source is a
real place-and-route, which is saga 6.

Both are load-bearing for step 8. A fit model that quietly assumed
values for them would produce numbers nobody could check, so
`gowin-profile` makes them unreadable as numbers and a test asserts
they stay that way -- if either is ever sourced, that test fails and
tells you to revisit `docs/fit-model.md`.

### Smaller gaps

The 9K's user I/O count is not stated as a number on its board page;
it needs Gowin's UG119E package and pinout guide. DSP counts for the
1K and 4K are likewise unstated. None of these block step 8, which
needs LUT4, flip-flops, block SRAM and DSP -- complete for the 9K,
asserted by a test.

**Which 25K board is on hand is unconfirmed.** Sipeed sells both a
Tang Nano 25K and a Tang Primer 25K. The fabric figures above are the
GW5A-25's, cited from the Primer 25K page; every board-level field is
`Unknown` rather than borrowed from the Primer.

The 9K remains primary because its open-toolchain support is the most
mature, which is what makes the fit predictions falsifiable against a
real place-and-route.

## 7. Saga roadmap

Saga 1 is created now. Later sagas are direction, not commitment; each
is defined properly when its predecessor's measurements land.

| Saga | Name | Goal |
| --- | --- | --- |
| 1 | `spm-walking-skeleton` | `.spm` format, seek-free stream, ternary GEMV reference, batch amortization measurement, Gowin profiles, fit report. One vertical slice on synthetic matrices. |
| 2 | `refine-the-fabric` | Three pieces of work, ordered by the evidence rather than by appeal -- see below. |
| 3 | `bitplane-and-q4` | Bit-serial / bitplane weight layout and Q4 datapath alongside ternary. Measure the silicon-area vs clock-cycles tradeoff the research calls out. |
| 4 | `tiny-model-import` | Import a real tiny model (TRM, ~7M params) to `.spm`. Layer-at-a-time correctness against a conventional implementation. |
| 5 | `recurrence-and-reuse` | Exploit TRM's recursion for temporal hardware reuse: circulating parameters, one small engine applied repeatedly. |
| 6 | `toolchain-crosscheck` | Hand-written RTL for the ternary lane, validated against saga 1 golden vectors; real Yosys/nextpnr fit diffed against the emulator's prediction on a Tang Nano 9K. |
| 7 | `moe-scheduling` | Parameter-centric (not token-centric) MoE scheduler: queue tokens per expert, scan each expert once. The research's likely killer application. |
| 8 | `multi-device` | Two-board pipeline and expert partitioning; the `Rp -> 0` claim at a scale that needs more than one part. |

### Saga 2, defined from what saga 1 measured

Three candidates, in the order the evidence supports. Each is sized to
answer a question saga 1 raised rather than to advance a plan.

**1. Overlapped fetch in `spm-stream-file`.** Its two buffer slots
refill synchronously, so `storage_time` and `compute_time` partition
wall clock and `eta` measures a serial pipeline. That is the single
caveat attached to every number in docs/results.md, and it is the
cheapest to remove: the seam is already there, and a prefetch thread
or io_uring backend drops in without touching `WeightStream` or any
consumer. Do this first because every later measurement inherits the
fix.

**2. Q4 and bitplane weight layouts.** Everything measured so far
applies to two-bit ternary alone, so every conclusion is really a
conclusion about BitNet-style models. Q4 widens that, and bitplane
layout is the one the research argues hardest for -- it trades silicon
area for clock cycles in a way that suits a fabric whose MAC array
only has to keep up with storage. The `.spm` encoding profile already
has room: it is one discriminant, and adding a value costs no format
break.

**3. A real tiny model, TRM (~7M parameters).** Synthetic matrices
have carried the format, the stream and both engines further than
expected, but they cannot say whether the layout survives contact with
real weight distributions, real shapes and a real quantization
pipeline. TRM's recursion also gives the circulating-parameter idea
something to exploit. Third because it is the largest piece and the
one that benefits most from the first two landing.

**Explicitly not saga 2: hardware.** The two figures that would
justify it -- bulk memory bandwidth and fabric fmax -- are still
`Unknown` for every board (section 6), and a test keeps them that way.
Buying or wiring anything before those are measured would be spending
on a guess.

Fronts 1 (rack Linux / DeepSeek-R1 1.58-bit) and 3 (RP2350 PIO) are out
of scope here, but this repository owns the `.spm` format and the golden
vectors both of them consume. That shared format is what makes the three
fronts one project rather than three disconnected experiments.

## 8. Saga 1 steps

Each step is one session, ends in a commit, and must leave `just check`
green.

1. **`repo-scaffold`** -- `components/` skeleton, `scripts/`
   (`serial.sh`, `gate.sh`, `check-locks.sh`, `check`, `build`),
   `justfile` delegating to them, `.cargo/config.toml` with a shared
   `target/`, `.gitignore` (including `.agentrail/sessions/`),
   `LICENSE`, `COPYRIGHT`, `README.md`, `CLAUDE.md` with the agentrail
   instruction block applied, and `docs/code_metrics.md` adapted from
   `../sw-mlpl`. Record the baseline `sw-checklist` counts.
   *Accepts when:* `just check` runs green on an otherwise empty tree
   and `sw-checklist` reports a recorded baseline.

2. **`spm-format`** -- `components/format/`: `spm-header`, `spm-codec`,
   `spm-layout`, `spm-file`. Ternary packing (2 bits per weight for
   {-1, 0, +1} plus one reserved code), per-group scales, stream
   directory, op descriptors.
   *Accepts when:* round-trip property tests pass for random matrices
   across group sizes and shapes, and the on-disk byte layout is
   pinned by a golden fixture.

3. **`spm-stream`** -- `components/stream/`: the `WeightStream` trait
   with no seek in its surface, memory and file implementations, and
   the metrics crate computing bandwidth, `eta`, `Ps`, `Rp`.
   *Accepts when:* a compile-fail test demonstrates that random access
   is not expressible through the trait, and metrics are exercised by
   unit tests with known-answer inputs.

4. **`spm-tensor-ref`** -- `components/tensor/`: accumulator banks with
   a batch dimension, ternary GEMV consuming a `WeightStream` via
   add / sub / skip with no multiplier, an f32 reference matmul, error
   metrics, and golden vector generation.
   *Accepts when:* ternary GEMV matches the f32 reference within stated
   tolerances on generated vectors, and the golden vectors serialize
   and reload identically.

5. **`spm-pack-cli`** -- `components/cli/`: `emufpga pack` converts a
   dense matrix to a ternary `.spm` in consumption order.
   *Accepts when:* `emufpga --help` and `emufpga --version` satisfy
   `sw-checklist` CLI validation, and packing then streaming a fixture
   reproduces the step 4 reference result byte for byte.

6. **`batch-amortization-bench`** -- `emufpga bench --batch 1,2,4,8,16,32`.
   This is the research's Experiment 1E in miniature and the
   make-or-break measurement: storage bandwidth stays roughly constant
   while useful operations rise with batch size until compute becomes
   limiting. That crossover is the number the whole architecture rests
   on.
   *Accepts when:* the crossover is measured and recorded in
   `docs/results.md`, with `Ps` rising approximately linearly in batch
   size below the crossover.

7. **`gowin-device-profiles`** -- `components/device/`: `gowin-profile`
   with the five Tang Nano boards. Every figure sourced and cited from
   Project Apicula or a Gowin datasheet; unsourceable figures recorded
   as unknown, never guessed.
   *Accepts when:* the section 6 table is replaced by cited values, the
   9K profile is complete, and a test asserts no profile field is a
   placeholder for the primary device.

8. **`resource-budget-and-fit`** -- **WITHDRAWN, not performed.**
   Specified as a per-board utilization and fit report. Withdrawn on a
   scope correction before any of it was written: we do not yet want
   high-fidelity FPGA emulation, and the two figures such a model most
   needs are `Unknown` for every board. Recorded in the saga rather
   than quietly rewritten.

9. **`conceptual-fabric-model`** -- `components/fabric/` and
   `emufpga sim`: a parameterized model with abstract knobs that
   executes a real `.spm` file, reports cycles, stalls and occupancy,
   and is **bit-exact** against `spm-gemv-ref`.
   *Accepts when:* agreement with the CPU reference is bit-exact (not
   within-tolerance -- in column-major order consecutive weights land
   on different accumulators, so summation order per accumulator is
   unchanged), a starved configuration reports high stalls, and cycle
   counts are monotone in the obvious directions.

10. **`saga-1-wrapup`** -- `README.md` results table, `docs/architecture.md`,
   `docs/spm-format.md`, final `sw-checklist` counts, saga 2 defined.
   *Accepts when:* `just check` is green, the format is documented well
   enough for the RP2350 and rack-Linux fronts to consume it, and
   `agentrail complete --done` closes the saga.

## 8a. Weights never enter git

GitHub hard-blocks a single file over 100 MB and wants a repository
under about 1 GB. Neither limit is the real danger: a large blob
entering history cannot be removed without rewriting it.

So model weights and everything derived from them stay out of the
tree. `.gitignore` covers `models/`, `*.pt`, `*.safetensors`, `*.ckpt`,
`*.gguf` and `*.spm` -- with an exception for the tiny golden fixtures
under `tests/golden/`, which are the format's byte contract and are
measured in bytes.

`scripts/check-size` enforces it as part of `just check`, failing on
any tracked or untracked file over 1 MiB. The limit is far below
GitHub's on purpose: nothing here has any business being megabytes,
and the useful place to stop a large file is before the commit rather
than after. Agentrail session transcripts are ignored for the same
reason -- one archived session was 3.1 MB, and ../sw-mlpl reports that
committing them repeatedly took its `.git` to 2.2 GB.

Weights are fetched or regenerated instead; `docs/` records where each
came from.

## 8b. What needs CUDA, and what does not

Nothing in this repository needs a GPU. `emufpga` is pure Rust with
one dependency (`clap`), the extractor is standard-library Python, and
the reference comparisons run on CPU or MPS. The Mac is sufficient for
the whole model ladder.

Three things do need CUDA, and they are worth cloning onto a
Linux/NVIDIA box rather than fighting on macOS.

### Runs fine without CUDA (done here)

| task | note |
| --- | --- |
| Everything in `components/` | pure Rust, no GPU path at all |
| `scripts/extract-checkpoint` | Rust; Python predecessor retained as a parity oracle |
| TRM reference comparison | torch CPU; verified at cosine 1.0 |
| HRM reference comparison | needs a `flash_attn` stand-in -- see below |

### Needs CUDA

**`flash-attn`, for running `sapientinc/HRM` unmodified.** Its
`models/layers.py` imports `flash_attn_func` at module scope with no
fallback, so the package must exist to import anything. Installing it
needs the CUDA toolkit and `nvcc`.

On the Mac this was worked around with a stand-in built on torch's
`scaled_dot_product_attention`. That substitution is sound for a
forward pass -- flash attention is a performance kernel computing
standard scaled dot-product attention, not a different function -- and
the comparison it enabled agreed at cosine 1.000000000000. It is
**not** a substitute for training, where flash-attn's memory behaviour
is the reason it exists.

**`adam-atan2`**, in HRM's requirements, is a CUDA extension. Only
training needs it.

**Training either model from scratch.** From the HRM repository's own
README: Sudoku-Extreme under 20 hours on one L40S, Maze-Hard under 24
hours on four L40S, ARC-AGI roughly three days on four H100. None of
that is reachable here, and a partially trained checkpoint would be
enough to exercise the streaming path anyway -- weight *distributions*
are what matter to it, not accuracy.

**A GPU baseline for the eventual comparison.** Measuring streamed
against conventional inference needs a real GPU on the conventional
side. That is rack work, not Mac work.

### If cloning onto Linux/NVIDIA

```
git clone https://github.com/sapientinc/HRM
pip install -r requirements.txt        # flash-attn, adam-atan2 need nvcc
```

Then `scratchpad/hrm/official.py` in this session runs unchanged with
the stand-in removed -- it only needs `sys.path` to stop pointing at
`shim/`. Worth re-running there once, to confirm the stand-in and the
real kernel agree; that closes the last assumption in the HRM
verification.

Checkpoints do not need CUDA to download or convert. The extractor
reads `.pt` and `.safetensors` with no torch at all.

## 9. Build and gate process

All build entry points live in `./scripts`; the `justfile` only
delegates. Mirrors `../sw-mlpl`.

| Script | Purpose |
| --- | --- |
| `scripts/serial.sh` | global build lock. Every component shares one `target/` dir, so two concurrent cargo invocations deadlock on the same build lock. Every cargo call routes through here. |
| `scripts/gate.sh <workspace> <pkg>...` | the pre-commit gate: `cargo metadata --locked`, `cargo fmt -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`, then `sw-checklist`. |
| `scripts/check-locks.sh [--fix]` | repo-wide `Cargo.lock` consistency sweep. A manifest change in one component strands the locks of every downstream workspace until someone builds there. |
| `scripts/check` | run `gate.sh` across the components a change touched. |
| `scripts/build` | build all components serially. |

### Checkpoint (`/mw-cp`)

Order matters and does not vary: **gates -> docs -> commit -> push ->
CHANGES.md refresh.** Under agentrail, `agentrail complete` comes AFTER
the commit lands, because the step records `HEAD` at completion time,
and nothing may change after `complete`.

- Scope tests to the components that changed; do not widen "just to be
  safe".
- All selected tests pass. Zero clippy warnings; fix, never suppress.
- `sw-markdown-checker -f "**/*.md"` when any `.md` changed. **Markdown
  in this repo is ASCII-only.**
- `sw-checklist` always, applying the ratchet policy: a commit should
  strictly lower both the failed and warning counts, or carry a
  `sw-checklist: exception` trailer justifying why not.
- Stage by explicit path, never `git add -A`. Include `.agentrail/`
  metadata in the same commit as the source it describes -- the most
  common agentrail failure mode is committing the source and forgetting
  the saga record, which leaves `agentrail audit` with a gap.
- Never `--no-verify`.

## 9a. Saga 1 final state

122 tests across six components. `sw-checklist`: **141 passed, 0
failed, 1 warning**, and no `#[allow]` anywhere in the tree -- every
clippy pedantic finding was retired by restructuring.

The standing warning is `Binary Freshness [emufpga]: emufpga is not
installed (run sw-install)`. It is not retired here on purpose: the
checkpoint process forbids running `sw-install` unasked, because it
would overwrite the stable installed binaries. Retiring it is a
deliberate act for whoever owns those binaries, not something a build
should do on its own.

The zero-failure, zero-`#[allow]` state was held from the first crate
rather than paid down, which was the goal set in
docs/code_metrics.md: it is far cheaper never to acquire the debt.

## 10. Risks and open questions

**Retired: the fit model is unfalsifiable.** This was the plan's
largest stated weakness -- a fit report producing numbers nobody could
check. It is retired by not building one. Step 8 was withdrawn and the
fabric model reports cycles, stalls and occupancy, all of which are
properties of a configuration rather than claims about a part. Nothing
in the tree now asserts anything a place-and-route could contradict,
which also means nothing in the tree is waiting on saga 6 to become
trustworthy.

**The conceptual model can drift toward false precision.** The failure
mode that replaces it. A cycle count looks like a measurement, and it
is easy to start reading occupancy as though it predicted silicon. Two
guards: cycles are never converted to a duration, and `gowin-profile`
has no consumer, so there is no path by which a device figure reaches
the model.

**Tang Nano bulk memory bandwidth may be the real ceiling.** The
architecture's entire premise is trading random access for cheap
sequential bandwidth. A 9K's PSRAM may not supply enough of it to make
even a tiny model interesting. Saga 1 step 8 should surface this early;
if it does, the finding is itself a result, and the response is to move
weight streaming off-chip (SPI flash arrays, host DMA, or the RP2350
front) rather than to abandon the premise.

**Ternary-only is a narrow starting datapath.** Saga 3 adds Q4 and
bitplane layouts. Until then, conclusions apply to
BitNet-style ternary models only, and the plan should not claim
otherwise.

**Unresolved, deferred to the sagas that need them:**

- Does the `.spm` format need a stable versioned wire format before the
  RP2350 and rack-Linux fronts start consuming it, or can it churn
  through saga 3? (Deferred to saga 1 step 9.)
- Which board hosts the first real RTL: the 9K for toolchain maturity,
  or the 20K for lane count? (Deferred to saga 6, decided by the step 8
  fit numbers.)
- Should the 4K's hard Cortex-M3 replace an external RP2350 as the
  stream controller, collapsing Front 3 onto one chip? (Deferred to
  saga 8.)
