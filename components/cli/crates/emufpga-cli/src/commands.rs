//! The subcommands' work, one function each.
//!
//! Each returns the text to print rather than printing it, so
//! integration tests can assert on the output without capturing
//! stdout.

use crate::run::Failure;
use spm_bench::run_sweep;
use spm_bench_report::render;
use spm_quantize::{Quantized, parse_matrix, quantize, write_spm};
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
