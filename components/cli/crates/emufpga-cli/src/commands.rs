//! The subcommands' work, one function each.
//!
//! Each returns the text to print rather than printing it, so
//! integration tests can assert on the output without capturing
//! stdout.

use crate::run::Failure;
use fabric_model::{FabricConfig, run_fabric};
use fabric_report::render as render_run;
use spm_activations::Activations;
use spm_bench::run_sweep;
use spm_bench_report::render;
use spm_quantize::{Quantized, parse_matrix, quantize, write_spm};
use spm_stream_mem::MemoryWeightStream;
use std::path::Path;

/// Reads a text matrix, quantizes it, and writes a `.spm` file.
///
/// # Errors
/// Returns [`Failure`] if the input cannot be read or parsed, or the
/// output cannot be written.
pub(crate) fn pack(input: &Path, output: &Path, group_size: u32) -> Result<String, Failure> {
    if group_size == 0 {
        return Err(Failure::new("--group-size must be at least 1"));
    }
    let text = std::fs::read_to_string(input)
        .map_err(|e| Failure::new(format!("cannot read {}: {e}", input.display())))?;
    let matrix =
        parse_matrix(&text).map_err(|e| Failure::new(format!("{}: {e}", input.display())))?;
    let quantized = quantize(&matrix, group_size);
    let bytes =
        write_spm(&quantized).map_err(|e| Failure::new(format!("cannot build .spm: {e}")))?;
    std::fs::write(output, &bytes)
        .map_err(|e| Failure::new(format!("cannot write {}: {e}", output.display())))?;
    Ok(summary(&quantized, bytes.len(), output))
}

/// Sweeps batch sizes against a `.spm` file and reports the metrics.
///
/// # Errors
/// Returns [`Failure`] if the batch list is empty or the file cannot
/// be read as a `.spm`.
pub(crate) fn bench(input: &Path, batches: &[usize], repeat: usize) -> Result<String, Failure> {
    if batches.is_empty() {
        return Err(Failure::new("--batch needs at least one size"));
    }
    if batches.contains(&0) {
        return Err(Failure::new("--batch sizes must be at least 1"));
    }
    let sweep = run_sweep(input, batches, repeat.max(1))
        .map_err(|e| Failure::new(format!("{}: {e}", input.display())))?;
    Ok(render(&sweep))
}

/// Runs a `.spm` file through the conceptual fabric model.
///
/// Activations are deterministic sixteenths -- exactly representable
/// in `f32`, so the run contributes no rounding of its own.
///
/// # Errors
/// Returns [`Failure`] if the file cannot be read, or the
/// configuration cannot describe a working pipeline.
pub(crate) fn sim(input: &Path, config: &FabricConfig, batch: usize) -> Result<String, Failure> {
    let bytes = std::fs::read(input)
        .map_err(|e| Failure::new(format!("cannot read {}: {e}", input.display())))?;
    let values: Vec<f32> = (0..4096)
        .map(|i| f32::from(u16::try_from(i % 33).unwrap_or(0)) / 16.0 - 1.0)
        .collect();
    let activations = Activations::broadcast(batch, &values);
    let outcome = run_fabric(MemoryWeightStream::new(bytes), &activations, config)
        .map_err(|e| Failure::new(format!("{}: {e}", input.display())))?;
    Ok(render_run(config, &outcome))
}

/// The line `pack` prints on success.
fn summary(quantized: &Quantized, bytes: usize, output: &Path) -> String {
    let d = &quantized.descriptor;
    format!(
        "packed {}x{} matrix, group size {}, {} groups, {bytes} bytes -> {}",
        d.rows,
        d.cols,
        d.group_size,
        quantized.scales.len(),
        output.display()
    )
}
