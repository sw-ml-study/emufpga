//! A run as markdown.

use fabric_model::{FabricConfig, FabricOutcome};
use std::fmt::Write;

/// Renders one run, with the caveat that makes it readable.
#[must_use]
pub fn render(config: &FabricConfig, outcome: &FabricOutcome) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "| metric | value |");
    let _ = writeln!(out, "| --- | ---: |");
    let _ = writeln!(out, "| weight lanes | {} |", config.weight_lanes);
    let _ = writeln!(out, "| batch width | {} |", config.batch_width);
    let _ = writeln!(out, "| fifo bytes | {} |", config.fifo_bytes);
    let _ = writeln!(
        out,
        "| fetch bytes/cycle | {} |",
        config.fetch_bytes_per_cycle
    );
    let _ = writeln!(out, "| weights | {} |", outcome.weights);
    let _ = writeln!(out, "| cycles | {} |", outcome.pipeline.cycles);
    let _ = writeln!(out, "| stall cycles | {} |", outcome.pipeline.stall_cycles);
    let _ = writeln!(
        out,
        "| backpressure cycles | {} |",
        outcome.pipeline.backpressure_cycles
    );
    out.push_str(&derived(outcome));
    out.push('\n');
    out.push_str(CAVEAT);
    out
}

/// The two ratios worth reading.
fn derived(outcome: &FabricOutcome) -> String {
    let mut out = String::new();
    let show = |value: Option<f64>, digits: usize| {
        value.map_or_else(|| "--".to_string(), |v| format!("{v:.digits$}"))
    };
    let _ = writeln!(
        out,
        "| occupancy | {} |",
        show(outcome.pipeline.occupancy(), 3)
    );
    let _ = writeln!(
        out,
        "| cycles per weight | {} |",
        show(outcome.cycles_per_weight(), 3)
    );
    out
}

/// What a run does and does not say.
const CAVEAT: &str = "\
Cycles are a UNIT, not a duration. No fabric clock has been measured,
so nothing here converts to seconds; multiply cycles per weight by a
clock period once someone measures one.

This is a conceptual model, not an FPGA simulator. It has no notion of
LUT4s, routing, clock domains or packing, and says nothing about
whether a design fits any part. Stalls mean the datapath waited on the
FIFO; backpressure means the fetch stage waited on the datapath. Both
large means the FIFO is too small to decouple them.

The arithmetic is not approximate: accumulators are bit-exact against
the CPU reference.
";
