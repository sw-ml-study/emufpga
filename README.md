# emufpga

A Serial Parameter Machine (SPM) research vehicle: a behavioral FPGA
emulator plus a resource budget, calibrated to the Gowin parts on Sipeed
Tang Nano boards, driving a streaming low-bit tensor engine.

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
- [docs/code_metrics.md](docs/code_metrics.md) -- the complexity gates

## License

MIT. See [LICENSE](LICENSE).
