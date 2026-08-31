//! Preparing a sweep: what shape is the file, and what goes in.

use spm_gemv_ref::GemvError;
use spm_stream_mem::MemoryWeightStream;

/// Reads the first descriptor's column count.
///
/// # Errors
/// Returns [`GemvError`] if the bytes are not a valid `.spm` or the
/// file declares no operations.
pub(crate) fn probe_cols(bytes: &[u8]) -> Result<usize, GemvError> {
    let groups = spm_stream_groups::GroupStream::open(MemoryWeightStream::new(bytes.to_vec()))?;
    let descriptor = groups.descriptors.first().ok_or(GemvError::NoStreams)?;
    Ok(descriptor.cols as usize)
}

/// A reproducible activation vector of `cols` sixteenths.
///
/// Exactly representable in `f32`, so the benchmark's inputs
/// contribute no rounding of their own and a rerun on another machine
/// sees the same arithmetic.
pub(crate) fn deterministic(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| f32::from(u16::try_from(i % 33).unwrap_or(0)) / 16.0 - 1.0)
        .collect()
}
