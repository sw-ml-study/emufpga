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
use spm_import::{assemble, parse_manifest, render_sidecar, total_weights};
use spm_order::apply_order;
use spm_quantize::{parse_matrix, quantize, write_spm};
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
    let d = &quantized.descriptor;
    Ok(format!(
        "packed {}x{} matrix, group size {}, {} groups, {} bytes -> {}",
        d.rows,
        d.cols,
        d.group_size,
        quantized.scales.len(),
        bytes.len(),
        output.display()
    ))
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

/// Converts an extracted checkpoint into a `.spm` plus its sidecar.
///
/// # Errors
/// Returns [`Failure`] if the manifest or a blob cannot be read, a
/// blob does not match its declared shape, or the output cannot be
/// written.
pub(crate) fn import(input: &Path, output: &Path, order: Option<&Path>) -> Result<String, Failure> {
    let manifest = input.join("manifest.tsv");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| Failure::new(format!("cannot read {}: {e}", manifest.display())))?;
    let listed = parse_manifest(&text).map_err(|e| Failure::new(e.to_string()))?;
    let (tensors, rotating) =
        apply_order(listed, order).map_err(|e| Failure::new(e.to_string()))?;
    let bytes = assemble(&tensors, input).map_err(|e| Failure::new(e.to_string()))?;
    std::fs::write(output, &bytes)
        .map_err(|e| Failure::new(format!("cannot write {}: {e}", output.display())))?;
    let sidecar = output.with_extension("names.tsv");
    let name = output.file_name().unwrap_or_default().to_string_lossy();
    std::fs::write(&sidecar, render_sidecar(&tensors, &name, rotating))
        .map_err(|e| Failure::new(format!("cannot write {}: {e}", sidecar.display())))?;
    Ok(format!(
        "imported {} tensors ({rotating} rotating), {} weights, {} bytes -> {} (+ {})",
        tensors.len(),
        total_weights(&tensors),
        bytes.len(),
        output.display(),
        sidecar.display()
    ))
}
