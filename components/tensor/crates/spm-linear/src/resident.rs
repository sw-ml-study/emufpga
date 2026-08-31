//! The conventional implementation: weights already in an array.

use core::fmt;

/// A matmul that could not be performed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearError {
    /// The weight count does not match the declared shape.
    WeightCount {
        /// Weights the shape implies.
        expected: usize,
        /// Weights supplied.
        found: usize,
    },
    /// An activation vector is shorter than the input dimension.
    ActivationLen {
        /// Inputs the shape implies.
        expected: usize,
        /// Inputs supplied.
        found: usize,
    },
    /// The parameter stream itself failed.
    Stream {
        /// What the stream layer said.
        detail: String,
    },
    /// The stream ended before the matrix did.
    Truncated {
        /// Weights the shape implies.
        expected: usize,
        /// Weights the stream supplied.
        found: usize,
    },
}

impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightCount { expected, found } => {
                write!(f, "shape implies {expected} weights, got {found}")
            }
            Self::ActivationLen { expected, found } => {
                write!(f, "need {expected} activations, got {found}")
            }
            Self::Stream { detail } => write!(f, "parameter stream: {detail}"),
            Self::Truncated { expected, found } => {
                write!(f, "stream ended after {found} of {expected} weights")
            }
        }
    }
}

impl std::error::Error for LinearError {}

/// Computes `Y = WX` with `W` resident, in column-major stream order.
///
/// Batched over `positions`: `activations` is `positions x cols` and
/// `out` is `positions x rows`, both row-major. One weight is applied
/// to every position before the next weight is looked at, which is the
/// same order the streamed version must use -- and the reason both can
/// agree bit for bit.
///
/// `weights[k]` is `W[k % rows][k / rows]`, the same order the stream
/// delivers -- so this reads the matrix exactly as the streamed
/// version does, and any disagreement is about mechanism rather than
/// about layout.
///
/// The inner loop is the same shape as the streamed one: hold the
/// activation for a column, walk down the rows. That is not an
/// accident of style; it is what makes bit-exact agreement possible.
///
/// # Errors
/// Returns [`LinearError`] if the weight count or an activation
/// vector disagrees with `(rows, cols)`.
pub fn resident(
    weights: &[f32],
    shape: (usize, usize),
    batch: (&[f32], usize),
    out: &mut [f32],
) -> Result<(), LinearError> {
    let (rows, cols) = shape;
    let (activations, positions) = batch;
    validate(weights.len(), shape, (activations.len(), positions))?;
    out[..positions * rows].fill(0.0);
    for (index, weight) in weights.iter().enumerate() {
        apply_weight(
            *weight,
            (index % rows, index / rows),
            (rows, cols, positions),
            activations,
            out,
        );
    }
    Ok(())
}

/// Checks a matmul's shapes before any arithmetic happens.
fn validate(
    weights: usize,
    shape: (usize, usize),
    supplied: (usize, usize),
) -> Result<(), LinearError> {
    let (rows, cols) = shape;
    let (activations, positions) = supplied;
    if weights != rows * cols {
        return Err(LinearError::WeightCount {
            expected: rows * cols,
            found: weights,
        });
    }
    if activations < positions * cols {
        return Err(LinearError::ActivationLen {
            expected: positions * cols,
            found: activations,
        });
    }
    Ok(())
}

/// Applies one weight to every batch position.
///
/// This is the reuse the architecture is built on: the weight was
/// fetched once and is used `positions` times before it is discarded.
pub(crate) fn apply_weight(
    weight: f32,
    at: (usize, usize),
    shape: (usize, usize, usize),
    activations: &[f32],
    out: &mut [f32],
) {
    let (row, col) = at;
    let (rows, cols, positions) = shape;
    for position in 0..positions {
        let slot = position * rows + row;
        out[slot] = weight.mul_add(activations[position * cols + col], out[slot]);
    }
}
