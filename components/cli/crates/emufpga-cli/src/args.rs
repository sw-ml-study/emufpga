//! Command line shape.

use clap::{Args, Parser, Subcommand};
use fabric_model::FabricConfig;
use std::path::PathBuf;

/// The `--version` block.
///
/// Shape follows the other Software Wrighter tools, and the fields are
/// the ones `sw-checklist` requires of a CLI: copyright, license,
/// repository, build host, build commit and build time. The build
/// fields come from `build.rs`.
const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n\nCopyright (c) 2026 Michael A Wright\nMIT License\n\n",
    "Repository: https://github.com/sw-ml-study/emufpga\n",
    "Build Host: ",
    env!("EMUFPGA_BUILD_HOST"),
    "\nBuild Commit: ",
    env!("EMUFPGA_BUILD_COMMIT"),
    "\nBuild Time: ",
    env!("EMUFPGA_BUILD_TIME"),
);

/// The `--help` body.
///
/// Longer than the one-line `about` shown by `-h`, and carrying the
/// agent section `sw-checklist` looks for. That section is not
/// ceremony: this repository is built step by step by agent sessions,
/// and the invariants below are the ones an agent most needs to not
/// get wrong.
const LONG_ABOUT: &str = "\
Serial Parameter Machine emulator and tools.

The SPM proposition inverts the usual accelerator memory model: put the
immutable weights in cheap sequential storage, move compute to the
weight stream, and keep only activations, accumulators and scales in
fast memory. This tool packs models into that layout and measures what
the trade buys.

AI CODING AGENT INSTRUCTIONS:

This tool operates on .spm files, a PHYSICAL EXECUTION LAYOUT rather
than a model interchange format. Weights are stored in exactly the
order the tensor engine consumes them, so reading a stream to the end
IS the matrix operation.

USAGE FOR AI AGENTS:
  1. `emufpga pack` converts a text matrix to .spm. Input is
     whitespace-separated f32, one matrix row per line; shape is
     inferred from the file.
  2. Quantization is per-group absmean (the BitNet b1.58 rule) and is
     lossy by design. Do not treat a round-trip through pack as
     value-preserving.
  3. --group-size sets weights per scale group in STREAM order.
     Setting it equal to the row count gives one scale per column,
     which lets the engine pre-scale the activation and keep its inner
     loop free of multipliers.

INVARIANTS THAT MUST NOT BE BROKEN:
  - The tensor engine never seeks into the parameter stream while
    performing an operation. Metadata and activations may live in
    ordinary RAM; the parameter stream may not.
  - The .spm byte layout is a contract shared with an RP2350 streamer
    and an FPGA loader. Changing it is a deliberate act: bump the
    version, regenerate the golden fixture, and say so.

Exit codes: 0 success, 1 the work failed, 2 the command line was wrong.

See docs/plan.md and docs/spm-format.md in the repository.";

/// Serial Parameter Machine tools: pack models, measure scans, and
/// check what fits on a Tang Nano board.
#[derive(Debug, Parser)]
#[command(
    name = "emufpga",
    version = VERSION,
    about = "Serial Parameter Machine emulator and tools",
    long_about = LONG_ABOUT
)]
pub struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// The subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Quantize a dense matrix to ternary and write a .spm file.
    ///
    /// Input is whitespace-separated f32, one matrix row per line;
    /// blank lines and `#` comments are ignored and the shape is
    /// inferred from the file. Weights are quantized with per-group
    /// absmean scaling and written in the column-major order the
    /// engine consumes.
    Pack {
        /// Text matrix to read.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        /// Where to write the .spm file.
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
        /// Weights per scale group, in stream order.
        ///
        /// Setting this equal to the row count gives one scale per
        /// column, which lets the engine pre-scale the activation once
        /// and keep its inner loop free of multipliers.
        #[arg(short, long, default_value_t = 64, value_name = "N")]
        group_size: u32,
    },

    /// Sweep batch sizes over a .spm file and report the scan metrics.
    ///
    /// Measures how far weight reuse goes before the tensor engine,
    /// rather than the parameter store, becomes the limit. Reports
    /// bandwidth, eta, scan productivity (Ps) and parameter residency
    /// (Rp) for each batch size, against both an in-memory store and
    /// a file.
    Bench {
        /// The .spm file to scan.
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        /// Batch sizes to sweep, comma separated.
        #[arg(
            short,
            long,
            value_name = "N,N,...",
            value_delimiter = ',',
            default_value = "1,2,4,8,16,32"
        )]
        batch: Vec<usize>,
        /// Passes per point. The fastest is reported and the spread
        /// shown, because a slow pass on a shared machine measures the
        /// scheduler rather than the engine.
        #[arg(short, long, default_value_t = 5, value_name = "N")]
        repeat: usize,
    },

    /// Run a .spm file through the conceptual fabric model.
    ///
    /// Reports cycles, stalls and occupancy for a configuration whose
    /// knobs are abstract -- lanes, FIFO depth, fetch rate -- and
    /// tied to no particular part. Cycles are a UNIT, not a duration:
    /// nothing here converts to seconds, because no fabric clock has
    /// been measured. This is conceptual exploration, not an FPGA
    /// simulator, and it says nothing about whether a design fits.
    Sim(SimArgs),

    /// Convert an extracted checkpoint into a .spm file.
    ///
    /// Input is what `scripts/extract-checkpoint` produces: a
    /// directory of raw little-endian f32 blobs and a manifest.tsv
    /// naming them. Those bytes are already the f32 encoding's wire
    /// format, so this is pure framing -- no weight value is read,
    /// reordered or rounded.
    ///
    /// Writes the .spm plus a sidecar name table beside it. The two
    /// belong together: .spm carries no names because the FPGA streams
    /// bytes in directory order and never needs them.
    Import {
        /// Directory holding manifest.tsv and the blobs.
        #[arg(short, long, value_name = "DIR")]
        input: PathBuf,
        /// Where to write the .spm file.
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,
        /// Consumption order specification.
        ///
        /// Without it the streams land in whatever order the manifest
        /// lists them, which for the extractor is alphabetical -- and
        /// alphabetical is NOT consumption order. A forward pass over
        /// such a file would have to seek backward, which this
        /// architecture forbids. See layouts/ for the shipped orders.
        #[arg(long, value_name = "FILE")]
        order: Option<PathBuf>,
    },
}

/// Knobs for `sim`.
///
/// Its own struct so the parsed flags become a [`FabricConfig`] in one
/// place, rather than being unpacked field by field at the dispatch
/// site. Note what is absent: there is no `--fmax` and no way to ask
/// for seconds. That is the guard against this becoming a fit model
/// by the back door, and a test asserts the flag stays unknown.
#[derive(Debug, Args)]
pub struct SimArgs {
    /// The .spm file to run.
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,
    /// Batch lanes.
    #[arg(short, long, default_value_t = 8, value_name = "N")]
    pub batch: usize,
    /// Weights the datapath consumes per cycle.
    #[arg(short = 'l', long, default_value_t = 8, value_name = "N")]
    pub weight_lanes: usize,
    /// Accumulator updates per weight per cycle.
    #[arg(short = 'w', long, default_value_t = 8, value_name = "N")]
    pub batch_width: usize,
    /// Weight FIFO depth in bytes.
    #[arg(short, long, default_value_t = 256, value_name = "BYTES")]
    pub fifo_bytes: usize,
    /// Bytes the parameter store delivers per cycle.
    #[arg(short = 'F', long, default_value_t = 16, value_name = "BYTES")]
    pub fetch_bytes_per_cycle: usize,
}

impl SimArgs {
    /// The fabric configuration these flags describe.
    #[must_use]
    pub fn config(&self) -> FabricConfig {
        FabricConfig {
            weight_lanes: self.weight_lanes,
            batch_width: self.batch_width,
            fifo_bytes: self.fifo_bytes,
            fetch_bytes_per_cycle: self.fetch_bytes_per_cycle,
            fetch_latency_cycles: 0,
        }
    }
}
