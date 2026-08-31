//! The same matmul, with `W` arriving off a stream.

use crate::resident::{LinearError, apply_weight};
use spm_codec_dense::decode_into;
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Computes `y = Wx`, pulling `W` off the next stream in `groups`.
///
/// Consumes exactly `rows * cols` weights, which is one whole stream,
/// and leaves `groups` positioned at the next one. That is the only
/// access pattern available: there is no way to ask for a stream by
/// index, because there is no seek. A caller runs streams in the order
/// the file declares them, which -- given a consumption-order layout
/// -- is the order the model wants.
///
/// Returns the index of the stream consumed, so a caller can check it
/// is where it thought it was.
///
/// The inner loop deliberately mirrors [`crate::resident`]: hold the
/// activation for a column, walk down the rows, and use `mul_add` in
/// both. Fused multiply-add rounds differently from a separate
/// multiply and add, so the two implementations have to agree on that
/// too or bit-exactness is lost for a reason that has nothing to do
/// with streaming.
///
/// # Errors
/// Returns [`LinearError::Truncated`] if the stream ends before the
/// matrix does, [`LinearError::Stream`] if the stream itself fails, or
/// [`LinearError::ActivationLen`] if too few activations were given.
pub fn streamed<S: WeightStream>(
    groups: &mut GroupStream<S>,
    shape: (usize, usize),
    batch: (&[f32], usize),
    out: &mut [f32],
) -> Result<usize, LinearError> {
    let (rows, cols) = shape;
    let (activations, positions) = batch;
    check(activations, positions * cols)?;
    let total = rows * cols;
    out[..positions * rows].fill(0.0);
    let (mut at, mut which, mut buffer) = (0usize, None, Vec::new());
    while at < total {
        let Some((stream, count)) = take(groups, &mut buffer)? else {
            return Err(LinearError::Truncated {
                expected: total,
                found: at,
            });
        };
        which.get_or_insert(stream);
        apply(&buffer, at, (rows, cols, positions), activations, out);
        at += count;
    }
    Ok(which.unwrap_or(0))
}

/// Rejects an activation vector shorter than the input dimension.
///
/// Checked before any group is pulled: the activations are resident
/// and known in advance, so a shape mismatch should never be
/// discovered part way through a stream that cannot be rewound
/// mid-operation.
fn check(activations: &[f32], cols: usize) -> Result<(), LinearError> {
    if activations.len() < cols {
        return Err(LinearError::ActivationLen {
            expected: cols,
            found: activations.len(),
        });
    }
    Ok(())
}

/// Pulls one group and decodes it into `buffer`.
///
/// `None` means the file ended, which the caller turns into a
/// truncation error against the shape it expected.
fn take<S: WeightStream>(
    groups: &mut GroupStream<S>,
    buffer: &mut Vec<f32>,
) -> Result<Option<(usize, usize)>, LinearError> {
    let Some(group) = groups.next_group() else {
        return Ok(None);
    };
    let group = group.map_err(|e| LinearError::Stream {
        detail: e.to_string(),
    })?;
    let count = group.count as usize;
    buffer.resize(count, 0.0);
    decode_into(group.packed, buffer).map_err(|needed| LinearError::Truncated {
        expected: needed,
        found: group.packed.len(),
    })?;
    Ok(Some((group.stream, count)))
}

/// Accumulates one group's weights, starting at stream position `at`.
///
/// Each weight is applied to every batch position before the next
/// weight is decoded -- the weight is fetched once and used
/// `positions` times, which is what `Ps` measures.
fn apply(
    weights: &[f32],
    at: usize,
    shape: (usize, usize, usize),
    activations: &[f32],
    out: &mut [f32],
) {
    let rows = shape.0;
    for (offset, weight) in weights.iter().enumerate() {
        let index = at + offset;
        apply_weight(
            *weight,
            (index % rows, index / rows),
            shape,
            activations,
            out,
        );
    }
}
