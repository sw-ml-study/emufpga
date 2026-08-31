# components/cli

The `emufpga` binary and the quantizer behind `pack`.

| Crate | Responsibility |
| --- | --- |
| `spm-quantize` | dense f32 matrix to ternary `.spm`, dependency-free |
| `spm-bench` | the batch-amortization sweep |
| `spm-bench-report` | rendering a sweep as markdown |
| `emufpga-cli` | argument parsing and subcommand dispatch |
| `emufpga` | thin binary: parse, dispatch, print, exit code |

`spm-quantize` stays free of external dependencies so saga 4's model
import can reuse the quantization rule without pulling a CLI parser
in. `emufpga-cli` uses `clap` -- the CLI grows to five subcommands and
`sw-checklist` has a check that looks for clap specifically.

Library and binary are split so integration tests call `run()`
directly. Shelling out would test the same code plus a process
boundary and make failures harder to read.

## The quantization rule

Per-group **absmean**, the rule BitNet b1.58 uses. For each scale group
in stream order:

```text
scale = mean(|w|) over the group
t     = clamp(round(w / scale), -1, +1)
```

Ties round half away from zero (`f32::round`). An all-zero group is
written as `scale = 1.0` with every weight `Zero`: that dequantizes
back to zero exactly and keeps a zero scale -- a value the hardware
would rather never see -- out of the wire format.

This is lossy, and the loss is the point of the architecture. It is
stated here and pinned by tests rather than hidden behind a default.

## Input format

Whitespace-separated f32, one matrix row per line. Blank lines and `#`
comments are ignored. Shape is inferred from the file rather than
passed as flags -- the file already knows its dimensions, and a
`--rows` that disagrees with the file is one more thing to get wrong.

Input is read row-major, the way a human writes a matrix. Output is
written column-major, the order the engine consumes. `quantize` is the
only place that transposition happens.

## Exit codes

| code | meaning |
|------|---------|
| 0 | success |
| 1 | the work failed (unreadable input, malformed matrix) |
| 2 | the command line itself was wrong (clap's convention) |

## `bench`

Sweeps batch sizes over a `.spm` file and reports bandwidth, `eta`,
`Ps` and `Rp` against both an in-memory store and a file. Results and
their caveats: [../../docs/results.md](../../docs/results.md);
reproduce with `just bench`.

Measurement and presentation are separate crates so they cannot grow
into each other, and because step 008's `fit` needs its own renderer.

`Crossover` is a three-armed enum -- `AlreadyBelow`, `At`,
`NotReached` -- because conflating "compute-bound before the sweep
began" with "compute became the limit at batch 8" would report a
measurement that was never taken. The CPU reference hits the first
case, and the report says `NOT MEASURED` rather than naming a batch
size.

Built by saga 1 steps 5 (spm-pack-cli) and 6 (batch-amortization-bench).
