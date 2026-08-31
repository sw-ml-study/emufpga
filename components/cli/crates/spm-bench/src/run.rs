//! Running the sweep.

use crate::model::{Backend, BenchRow, Sweep, timer_overhead};
use crate::setup::{deterministic, probe_cols};
use spm_activations::Activations;
use spm_gemv_ref::{GemvError, run_gemv};
use spm_stream_file::FileWeightStream;
use spm_stream_mem::MemoryWeightStream;
use std::path::Path;
use std::time::{Duration, Instant};

/// Sweeps `batches` against both backends, `repeat` passes each.
///
/// # Errors
/// Returns [`GemvError`] if the file cannot be read or is not a valid
/// `.spm`.
pub fn run_sweep(path: &Path, batches: &[usize], repeat: usize) -> Result<Sweep, GemvError> {
    let bytes = std::fs::read(path).map_err(spm_stream::StreamError::Io)?;
    let cols = probe_cols(&bytes)?;
    let activations = deterministic(cols);
    let mut rows = Vec::new();
    for backend in [Backend::Memory, Backend::File] {
        for &batch in batches {
            rows.push(measure(
                path,
                &bytes,
                backend,
                (batch, repeat),
                &activations,
            )?);
        }
    }
    Ok(Sweep {
        rows,
        timer_overhead: timer_overhead(),
        repeat: repeat.max(1),
    })
}

/// Runs one (backend, batch) point `repeat` times, keeping the best.
///
/// Fastest pass rather than mean: a slow pass on a shared machine
/// measures the scheduler, not the engine. The spread goes into the
/// row so a reader can see how much that choice mattered.
fn measure(
    path: &Path,
    bytes: &[u8],
    backend: Backend,
    plan: (usize, usize),
    activations: &[f32],
) -> Result<BenchRow, GemvError> {
    let (batch, repeat) = plan;
    let lanes = Activations::broadcast(batch, activations);
    let (best, fastest, slowest) = best_pass(path, bytes, backend, (&lanes, repeat))?;
    Ok(BenchRow {
        backend,
        batch,
        best,
        fastest,
        slowest,
    })
}

/// Repeats a scan, returning the fastest pass and the time bounds.
fn best_pass(
    path: &Path,
    bytes: &[u8],
    backend: Backend,
    plan: (&Activations, usize),
) -> Result<(spm_stream_metrics::ScanMetrics, Duration, Duration), GemvError> {
    let (lanes, repeat) = plan;
    let mut best = None;
    let (mut fastest, mut slowest) = (Duration::MAX, Duration::ZERO);
    for _ in 0..repeat.max(1) {
        let started = Instant::now();
        let metrics = scan_once(path, bytes, backend, lanes)?;
        let elapsed = started.elapsed();
        slowest = slowest.max(elapsed);
        if elapsed < fastest {
            (fastest, best) = (elapsed, Some(metrics));
        }
    }
    Ok((best.unwrap_or_default(), fastest, slowest))
}

/// One scan against one backend.
fn scan_once(
    path: &Path,
    bytes: &[u8],
    backend: Backend,
    lanes: &Activations,
) -> Result<spm_stream_metrics::ScanMetrics, GemvError> {
    let outcome = match backend {
        Backend::Memory => run_gemv(MemoryWeightStream::new(bytes.to_vec()), lanes)?,
        Backend::File => run_gemv(FileWeightStream::open(path)?, lanes)?,
    };
    Ok(outcome.metrics)
}
