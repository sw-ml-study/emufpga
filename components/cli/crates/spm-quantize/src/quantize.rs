//! Row-major f32 in, stream-order ternary out.

use crate::matrix::{Matrix, to_stream_order};
use spm_codec::Ternary;
use spm_layout::{Encoding, OpDescriptor, group_count, group_len, weight_count};

/// A quantized matrix, ready to write.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantized {
    /// Shape and scale-group size.
    pub descriptor: OpDescriptor,
    /// Weights in stream order (column-major).
    pub weights: Vec<Ternary>,
    /// One scale per group, in stream order.
    pub scales: Vec<f32>,
}

/// Quantizes `matrix` with per-group absmean scaling.
///
/// The transposition from row-major input to column-major stream order
/// happens here, and only here.
///
/// # Panics
/// Panics if the matrix's declared shape disagrees with its value
/// count, which would be a parser bug rather than bad input.
#[must_use]
pub fn quantize(matrix: &Matrix, group_size: u32) -> Quantized {
    let rows = u32::try_from(matrix.rows).unwrap_or(u32::MAX);
    let cols = u32::try_from(matrix.cols).unwrap_or(u32::MAX);
    let total = weight_count(rows, cols);
    let (weights, scales) = quantize_groups(&to_stream_order(matrix), total, group_size);
    Quantized {
        descriptor: OpDescriptor {
            rows,
            cols,
            group_size,
            encoding: Encoding::Ternary2F32I32,
            lane_count: 1,
        },
        weights,
        scales,
    }
}

/// Quantizes stream-order values group by group.
///
/// Each group gets its own absmean scale, so a group of small weights
/// keeps its resolution instead of being flattened by a loud
/// neighbour elsewhere in the matrix.
fn quantize_groups(stream: &[f32], total: u64, group_size: u32) -> (Vec<Ternary>, Vec<f32>) {
    let groups = group_count(total, group_size);
    let mut weights = Vec::with_capacity(stream.len());
    let mut scales = Vec::with_capacity(usize::try_from(groups).unwrap_or(0));
    let mut at = 0usize;
    for group in 0..groups {
        let len = group_len(total, group_size, group) as usize;
        let scale = absmean(&stream[at..at + len]);
        weights.extend(stream[at..at + len].iter().map(|w| to_ternary(*w, scale)));
        scales.push(scale);
        at += len;
    }
    (weights, scales)
}

/// The mean of `|w|` over a group, or `1.0` for an all-zero group.
///
/// An all-zero group has no meaningful scale. `1.0` dequantizes back
/// to zero exactly, and keeps a zero scale -- a value the hardware
/// would rather never see -- out of the wire format.
fn absmean(group: &[f32]) -> f32 {
    if group.is_empty() {
        return 1.0;
    }
    let total: f32 = group.iter().map(|w| w.abs()).sum();
    let count = f32::from(u16::try_from(group.len()).unwrap_or(u16::MAX));
    let mean = total / count;
    if mean > 0.0 { mean } else { 1.0 }
}

/// Rounds one weight to `-1`, `0` or `+1` against its group scale.
///
/// `f32::round` breaks ties away from zero, so `0.5 * scale` becomes
/// `+1` rather than `0`.
fn to_ternary(weight: f32, scale: f32) -> Ternary {
    match (weight / scale).round() {
        v if v >= 1.0 => Ternary::Plus,
        v if v <= -1.0 => Ternary::Minus,
        _ => Ternary::Zero,
    }
}
