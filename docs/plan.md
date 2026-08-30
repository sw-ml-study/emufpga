# emufpga -- implementation plan

Serial Parameter Machine (SPM) research vehicle: a behavioral FPGA
emulator plus resource budget, calibrated to the Gowin parts on Sipeed
Tang Nano boards, driving a streaming low-bit tensor engine.

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

**emufpga is** a cycle-accounting behavioral model of an SPM streaming
tensor engine, paired with a resource budget checked against real Gowin
device profiles. It answers two questions for a proposed engine
configuration:

1. Is it correct? (bit-exact against a CPU reference over identical
   golden vectors)
2. Would it fit, and how fast would it run, on the Tang Nano board in
   my hand?

**emufpga is not** a gate-level or bitstream-accurate Gowin simulator,
and it does not emit HDL. RTL is written by hand later and validated
against the golden vectors this repository produces. This boundary is
deliberate: it keeps the research question (does sequential parameter
access plus reuse do useful work?) ahead of the toolchain question.

## 2. Decisions taken

| Question | Decision |
| --- | --- |
| Emulator depth | Behavioral cycle model + resource budget. No HDL emission. |
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

**`components/stream/`** -- sequential parameter access

| Crate | Responsibility |
| --- | --- |
| `spm-stream` * | `WeightStream` trait, block and FIFO types, seek-free by construction |
| `spm-stream-mem` * | in-memory impl, used by tests |
| `spm-stream-file` * | buffered/double-buffered file impl |
| `spm-stream-metrics` * | bandwidth, `eta`, `Ps`, `Rp` accounting |

**`components/tensor/`** -- CPU golden reference

| Crate | Responsibility |
| --- | --- |
| `spm-accum` * | accumulator banks, batch lanes |
| `spm-gemv-ref` * | reference ternary/Q4 GEMV over a `WeightStream` (add / sub / skip, no multiplier) |
| `spm-numeric` * | f32 reference matmul, error metrics |
| `spm-vectors` * | golden vector generation and serialization |

**`components/device/`** -- Gowin device profiles

| Crate | Responsibility |
| --- | --- |
| `gowin-profile` * | profile data types, the Tang Nano board table |
| `gowin-budget` * | LUT4/DFF/BSRAM/DSP/IO accounting, `fits()` |
| `gowin-timing` * | fmax model, cycles -> seconds |

**`components/fabric/`** -- behavioral cycle model (saga 2)

| Crate | Responsibility |
| --- | --- |
| `emufpga-rtl` | behavioral blocks (FIFO, decoder, MAC lane, accumulator bank), each declaring its own resource cost |
| `emufpga-pipeline` | compose blocks into a pipeline, elaborate |
| `emufpga-clock` | cycle-driven scheduler |
| `emufpga-trace` | event/waveform trace for debugging |

**`components/engine/`** -- SPM engine on the fabric (saga 2)

| Crate | Responsibility |
| --- | --- |
| `spm-engine` | streaming ternary GEMV expressed as a fabric pipeline |
| `spm-engine-config` | lane count, batch width, datapath width, derivation |
| `spm-verify` | differential test: fabric vs `spm-gemv-ref` |

**`components/report/`** -- reporting (saga 2)

| Crate | Responsibility |
| --- | --- |
| `emufpga-fit` | utilization report per device |
| `emufpga-perf` | throughput and scan-productivity report |
| `emufpga-render` | text / markdown / TSV rendering |

**`components/cli/`** -- binaries

| Crate | Responsibility |
| --- | --- |
| `emufpga-cli` * | argument parsing and subcommand dispatch |
| `emufpga` * | thin binary; must satisfy `sw-checklist` help/version validation |

Subcommands: `pack`, `bench`, `fit`, `sim`, `verify`.

## 6. Device profiles

Target boards, all owned. **These figures are UNVERIFIED starting
points.** Saga 1 step 7 replaces them with values sourced and cited
from Project Apicula's device database and the Gowin datasheets; any
number that cannot be sourced is recorded as unknown rather than
guessed.

| Board | Device | LUT4 | BSRAM | DSP | On-board bulk memory | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Tang Nano 1K | GW1NZ-1 | ~1152 | ~72 Kb | 0 | none | low power; smallest useful fabric |
| Tang Nano 4K | GW1NSR-4C | ~4608 | ~180 Kb | ~16 | HyperRAM | hard Cortex-M3; can play the stream-controller role on one chip |
| **Tang Nano 9K** | **GW1NR-9** | **~8640** | **~468 Kb** | **~20** | **PSRAM** | **primary target** |
| Tang Nano 20K | GW2AR-18 | ~20736 | ~828 Kb | ~48 | SDRAM | most lanes of the pre-25K parts |
| Tang Nano 25K | GW5A-25 | ~23000 | TBD | TBD | TBD | Arora V; least mature open-toolchain support |

A profile records at minimum: LUT4 count, DFF count, BSRAM block count
and block width, DSP block count and shape, user IO count, achievable
fmax band, and the type/width/bandwidth of on-board bulk memory. The
last field matters most -- it sets the parameter-stream bandwidth the
whole architecture is bottlenecked on.

The 9K is primary because its open-toolchain support is the most
mature, which is precisely what makes the emulator's fit predictions
falsifiable: `just fit` can eventually run a real nextpnr place-and-route
and diff utilization against the emulator's claim. A fit model nobody
can check is not worth building.

## 7. Saga roadmap

Saga 1 is created now. Later sagas are direction, not commitment; each
is defined properly when its predecessor's measurements land.

| Saga | Name | Goal |
| --- | --- | --- |
| 1 | `spm-walking-skeleton` | `.spm` format, seek-free stream, ternary GEMV reference, batch amortization measurement, Gowin profiles, fit report. One vertical slice on synthetic matrices. |
| 2 | `fabric-cycle-model` | `components/fabric/` + `components/engine/`: behavioral blocks with declared resource cost, cycle scheduler, differential verification against the saga 1 reference. |
| 3 | `bitplane-and-q4` | Bit-serial / bitplane weight layout and Q4 datapath alongside ternary. Measure the silicon-area vs clock-cycles tradeoff the research calls out. |
| 4 | `tiny-model-import` | Import a real tiny model (TRM, ~7M params) to `.spm`. Layer-at-a-time correctness against a conventional implementation. |
| 5 | `recurrence-and-reuse` | Exploit TRM's recursion for temporal hardware reuse: circulating parameters, one small engine applied repeatedly. |
| 6 | `toolchain-crosscheck` | Hand-written RTL for the ternary lane, validated against saga 1 golden vectors; real Yosys/nextpnr fit diffed against the emulator's prediction on a Tang Nano 9K. |
| 7 | `moe-scheduling` | Parameter-centric (not token-centric) MoE scheduler: queue tokens per expert, scan each expert once. The research's likely killer application. |
| 8 | `multi-device` | Two-board pipeline and expert partitioning; the `Rp -> 0` claim at a scale that needs more than one part. |

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

8. **`resource-budget-and-fit`** -- `gowin-budget` and `gowin-timing`;
   `emufpga fit` reports LUT4 / BSRAM / DSP / IO utilization plus
   predicted cycles and throughput per board for a given lane and batch
   configuration.
   *Accepts when:* `emufpga fit` prints a per-board table for the
   synthetic engine, correctly refuses configurations that exceed a
   profile, and its assumptions are written down in `docs/fit-model.md`
   so a later real place-and-route can contradict them specifically.

9. **`saga-1-wrapup`** -- `README.md` results table, `docs/architecture.md`,
   `docs/spm-format.md`, final `sw-checklist` counts, saga 2 defined.
   *Accepts when:* `just check` is green, the format is documented well
   enough for the RP2350 and rack-Linux fronts to consume it, and
   `agentrail complete --done` closes the saga.

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

## 10. Risks and open questions

**The fit model is unfalsifiable until saga 6.** Between now and then,
`emufpga fit` produces numbers nobody has checked against a real
place-and-route. Mitigation: `docs/fit-model.md` states every assumption
explicitly so saga 6 can contradict specific claims rather than shrug at
the whole model. This is the plan's largest weakness and it is accepted
deliberately, because the alternative -- building the toolchain
integration first -- delays the research question by months.

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
