//! The `pack` subcommand.

use crate::run::Failure;
use spm_quantize::{Quantized, parse_matrix, quantize, write_spm};
use std::path::Path;

/// Reads a text matrix, quantizes it, and writes a `.spm` file.
///
/// Returns a one-line summary for the caller to print, so the work is
/// testable without capturing stdout.
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
