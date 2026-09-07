# emufpga

A Serial Parameter Machine (SPM) research vehicle: a behavioral FPGA
emulator plus a resource budget, calibrated to the Gowin parts on Sipeed
Tang Nano boards, driving a streaming low-bit tensor engine.

**[Open the live serial-MoE FPGA visual emulator](https://sw-ml-study.github.io/emufpga/)**

[![Serial Parameter Machine research notebook comparing resident GPU weights with a serial expert stream](docs/assets/serial-moe-fpga-demo.png)](https://sw-ml-study.github.io/emufpga/)

The live view animates the purpose-built datapath and presents graphical
memory, traffic, and throughput comparisons with measured/derived/projected
provenance, plus switchable animated experiment diagrams spanning HDD, NVMe,
PMEM, CPU/DRAM, GPU/VRAM, FPGA, and MCU/PIO. Click the image to interact with
schedules, context sizes, and proposed data flows.

## The idea

Conventional inference treats gigantic immutable weight matrices as
though they need general-purpose random-access memory, even though the
dominant operation traverses them in a fully predictable order. The SPM
proposition inverts that: put the weights in cheap sequential storage,
move compute to the weight stream, and keep only activations,
accumulators, scales and recurrent state in fast memory.

Never ask the parameter store for an arbitrary weight. Arrange weights
physically in exactly the order the tensor engine consumes them, then
start a scan. When the stream reaches the end of the layer, the matrix
operation is finished.

**The goal is not speed.** It is doing more with less at equal
correctness: less VRAM, less system RAM, older and cheaper hardware,
fewer kWh. On a 135M model, streaming holds **4 KiB** of weights
resident instead of 269 MB, gives bit-exact answers, and serves five
clients asking five different questions off one pass of the weights.
Projected onto a 300 GB mixture-of-experts model, that is about 1.4 GB
of RAM for five clients instead of 300 GB of VRAM.

**See it run in your browser:** paste
[`for-mlpl-playground-editor/serial-parameter-machine.mlpl`](for-mlpl-playground-editor/serial-parameter-machine.mlpl)
into the [MLPL playground](https://sw-ml-study.github.io/sw-mlpl/).
Self-contained, with charts and diagrams, and every claim either
computed live or measured on a real model.

Moving to another machine:
[docs/handoff-linux-cuda.md](docs/handoff-linux-cuda.md) -- what to
fetch, what needs CUDA, and what is queued.

Start here: [docs/why-this-saves-ram.md](docs/why-this-saves-ram.md) --
the plain-language case, what has actually been measured, why MoE is
the best case rather than a stretch goal, and what is not yet true.

The shortest falsifiable status is
[docs/claim-scorecard.md](docs/claim-scorecard.md): today there is **no measured
agents-per-kWh advantage**; it defines the workload and success, failure, and
mixed-result thresholds needed to make that claim.

Newest result: the oversized Gemma-4 Q5_K_M artifact now completes end-to-end
generation with native GPU attention/state and reclaimable CPU selected-expert
pages. The three-run short smoke passed 45/45 tasks at 4.17 GiB peak VRAM;
all 262,144 final logits for one complete prompt were bit-identical with and
without per-expert page reclamation. A residency audit found the 13.98 GiB
peak RSS was 13.17 GiB file-backed mappings and only 0.80 GiB anonymous memory.
This is a capacity success and one-path arithmetic check. A follow-up paired
Rust corpus compiled and executed generated functions in isolated containers:
both resident and reclaimed policies passed 30/30, with zero outcome
disagreements. That is bounded executable reliability evidence, not validation
of repository-scale coding agents, which remains open.
See [docs/gemma4-serial-experts.md](docs/gemma4-serial-experts.md).

The first direct multi-agent comparison now measures the project's actual
sharing target. Four held-out Rust tasks, repeated three times, passed 12/12
both sequentially and at concurrency four with exactly 690 completion tokens
per policy. Existing batched expert execution reduced logical expert traversal
by 25.2%, finished inference 33.9% sooner, and raised aggregate token
throughput by 51.4%. Eight agents gained another 19.6% over two matched c4
batches while using 17.4% fewer logical expert bytes per routed assignment.
This is useful but strongly diminishing reuse, not linear scaling,
physical-storage savings, or an FPGA result. See
[docs/workload-aware-expert-cache.md](docs/workload-aware-expert-cache.md).

The new model-install phase makes that cache work repeatable rather than
Gemma-specific. It safely inventories routed GGUF tensors, binds the result to
the exact model/runtime/hardware/budgets, and records trace frequency,
co-activation, reuse, and held-out ranking stability. Granite demonstrates a
graceful no-prior cold start; Gemma demonstrates a workload-derived proposal.
The placement is not yet a measured speedup. See
[docs/model-install-expert-profile.md](docs/model-install-expert-profile.md).

The first physical use of that frozen profile is a useful negative result.
Across 48 cold expert-byte replays, static preload hit 12.6-15.9% but increased
physical reads 8.4-9.8%; the wider warm tier hit 23.4-27.6% but increased reads
18.4-22.0%. All byte-path digests agreed. Do not confuse multi-agent sharing,
which remains measured, with speculative preload, which did not repay its cold
cost. See [docs/profile-driven-cache-results.md](docs/profile-driven-cache-results.md).

A 12-batch persistent-cache follow-up finds the first limited win: 128 MiB
admit-on-second-use avoids 11.6% of userspace model copying and breaks even by
batch two on repeated held-out routing. It saves zero physical HDD bytes because
Linux page cache already retains the file, and 512 MiB buys no extra hits. See
[docs/demand-cache-break-even.md](docs/demand-cache-break-even.md).

With model pages forcibly evicted between batches, the same 128 MiB cache does
save storage traffic: 4.8% +/-2.2 percentage points physical bytes across three
six-batch runs, breaking even by batch two. This is controlled expert-byte
replay, not inference. See
[docs/direct-cache-break-even.md](docs/direct-cache-break-even.md).


The completed cold-HDD qualification is recorded in
[docs/intermediate-results.md](docs/intermediate-results.md). The ordinary
loader read the complete 19.32 GB model before serving; lazy expert marking
reduced startup reads to 3.83 GB. Three repetitions at concurrency 1/2/4/8
passed all 180 executable Rust tasks. Eight tasks read 10.70 GB versus 8.12 GB
for one, improving resident completion rate from 23.2 to 109.8 tasks/hour.
Reclamation saved RAM but did not reduce physical bytes. Reads remained nearly
all 4 KiB page faults: a negative mmap control, not a sequential-stream test.
The next storage experiment compares it with bounded large reads from one and
multiple SAS disks plus an SSD hot tier.

Background and the full argument: [docs/research.txt](docs/research.txt).

## What this is

A conceptual, cycle-approximate model of an SPM streaming tensor
engine. It answers two questions about a proposed engine
configuration:

1. **Is it correct?** Bit-exact against a CPU reference over identical
   golden vectors.
2. **Where does the pipeline stall?** Does the datapath starve waiting
   on the parameter stream, or does the stream back up waiting on the
   datapath?

Its knobs are abstract -- lanes, FIFO depth, fetch rate -- and cycles
are a unit, not a duration.

## What this is not

Not a gate-level or bitstream-accurate Gowin simulator, it emits no
HDL, and it does **not** predict whether a design fits a given part.
No LUT4 budgets, no utilization percentages, no place-and-route
predictions. RTL is written by hand later and validated against the
golden vectors this repository produces.

Something FPGA-like that can be refined later beats something
device-shaped that cannot be checked. A resource-budget and fit report
was planned and withdrawn for exactly that reason -- see docs/plan.md
section 1.

## Target hardware

Primary: **Tang Nano 9K (Gowin GW1NR-9)**, chosen because its
open-toolchain support (Yosys, nextpnr-himbaechel, Project Apicula,
openFPGALoader) is the most mature, which is what makes the emulator's
fit predictions falsifiable against a real place-and-route. Profiles for
the 1K, 4K, 20K and 25K boards are data, not new code.

## Metrics

Raw GB/s alone does not express what this architecture improves, so
every benchmark also reports:

- **`eta`** -- engine consumption bandwidth / storage bandwidth. Can
  compute keep up with storage? If not, the hardware design has its
  target.
- **`Ps`** (scan productivity) -- useful parameter applications per
  parameter value read. Batch 1 gives `Ps ~= 1`; batching, MoE
  scheduling and speculative decoding raise it. This is the entire
  economic argument.
- **`Rp`** (parameter residency) -- parameter bytes resident in RAM over
  total parameter bytes. Conventional inference is ~1. The goal is
  `Rp -> 0` while activation RAM stays nonzero.

## Layout

No root `Cargo.toml`. Every directory under `components/` is its own
cargo workspace -- the horizontal spread is what keeps the code inside
the complexity gates in [docs/code_metrics.md](docs/code_metrics.md).

All build entry points live in `./scripts`. Because every workspace
shares one `target/` dir, concurrent cargo invocations deadlock on the
same build lock, so every cargo call routes through
`scripts/serial.sh`.

```
just check        # pre-commit gate over changed components
just check-all    # ... over every component
just build        # build all components, serially
just bench        # reproduce the docs/results.md measurement
just locks        # Cargo.lock consistency sweep
just size         # fail on files too large for git history
```

```
emufpga pack  -i matrix.txt -o model.spm -g 64
emufpga bench -i model.spm -b 1,2,4,8,16,32
emufpga sim   -i model.spm -l 8 -w 8 -f 256 -F 64
```

## Results

Saga 1 measured the make-or-break experiment. The headline is a
negative result with a usable number attached.

| what | measured |
| --- | --- |
| Scan productivity `Ps` | exactly the batch size, 1 to 128 |
| Storage traffic across batch sizes | unchanged |
| Parameter residency `Rp` | 0.00195 -- one group buffer, not one model |
| CPU engine vs a page-cached read | **~196x too slow to saturate it** |
| Fabric model, 8 lanes | store-bound/compute-bound crossing at 4-16 bytes/cycle |

Batching works: 128x the batch buys 44x aggregate throughput while the
store does no more work. But the crossover the experiment went looking
for -- where compute stops keeping up with storage -- **is not in
range**. The engine is compute-bound at every batch size, so the
crossing lies below the smallest one measured. The tool reports that
as `NOT MEASURED` rather than naming a batch size it did not observe.

Both models agree on the direction: arithmetic is the scarce resource,
not bytes.

**What these numbers do not support:** IO is not overlapped, so `eta`
measures a serial pipeline rather than anything an FPGA would do; the
store was warm, making 196x a *lower* bound on the shortfall; the
engine is unoptimised scalar reference code; and it is one matrix on
one machine. Fabric cycles are a unit, not a duration -- nothing here
converts to seconds, because no fabric clock has been measured.

Full numbers, method and caveats: [docs/results.md](docs/results.md).
Reproduce with `just bench`.

## Status

Saga 1 (`spm-walking-skeleton`) complete: the `.spm` format, a
seek-free weight stream, the multiplier-free ternary GEMV reference,
sourced Gowin device profiles (reference data, no consumer), the
conceptual fabric model, and `emufpga pack` / `bench` / `sim`.

122 tests across six components. One planned step was withdrawn: a
resource-budget and fit report, dropped on a scope correction because
device-accurate fit modelling is not what this project wants yet, and
would have been unfalsifiable anyway.

- [docs/architecture.md](docs/architecture.md) -- how the pieces fit
- [docs/plan.md](docs/plan.md) -- decisions, roadmap, risks
- [docs/spm-format.md](docs/spm-format.md) -- the wire contract
- [docs/results.md](docs/results.md) -- what was measured
- [docs/storage-tier-break-even.md](docs/storage-tier-break-even.md) -- HDD,
  NVMe, cache, and overlap ceilings for Granite expert streams
- [docs/async-prefetch-validation.md](docs/async-prefetch-validation.md) --
  observed bounded-prefetch correctness and mixed speed results
- [docs/code_metrics.md](docs/code_metrics.md) -- the complexity gates
- [docs/granite-moe-serial.md](docs/granite-moe-serial.md) -- serial MoE expert verification
- [docs/moe-memory-economics.md](docs/moe-memory-economics.md) -- MoE memory and bandwidth ledger
- [docs/moe-q6-benchmark.md](docs/moe-q6-benchmark.md) -- packed Q6_K MoE benchmark
- [docs/moe-full-packed-forward.md](docs/moe-full-packed-forward.md) -- full chained packed-MoE validation
- [docs/moe-routing-distributions.md](docs/moe-routing-distributions.md) -- empirical routing unions and repeated timing distributions
- [docs/moe-rtl-cosim.md](docs/moe-rtl-cosim.md) -- hardware-shaped Q6_K cycle validation
- [docs/analyze-HLS4ML.md](docs/analyze-HLS4ML.md) -- what published HLS/FPGA findings do and do not validate here
- [docs/xeon-arch-considerations.md](docs/xeon-arch-considerations.md) -- Xeon generations, NUMA, PMEM, and expert-worker scheduling
- [live visualization](https://sw-ml-study.github.io/emufpga/) ([local instructions](visualization/README.md)) -- animated serial-MoE FPGA datapath
- [docs/serial-processing-eli5.md](docs/serial-processing-eli5.md) -- what else can stream and complementary approaches
- [docs/postmortem-1.md](docs/postmortem-1.md) -- what went wrong, and what caught it

## License

MIT. See [LICENSE](LICENSE).
