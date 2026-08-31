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

/// Computes `y = Wx` with `W` resident, in column-major stream order.
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
    activations: &[f32],
    out: &mut [f32],
) -> Result<(), LinearError> {
    let (rows, cols) = shape;
    if weights.len() != rows * cols {
        return Err(LinearError::WeightCount {
            expected: rows * cols,
            found: weights.len(),
        });
    }
    if activations.len() < cols {
        return Err(LinearError::ActivationLen {
            expected: cols,
            found: activations.len(),
        });
    }
    out[..rows].fill(0.0);
    for (index, weight) in weights.iter().enumerate() {
        out[index % rows] = weight.mul_add(activations[index / rows], out[index % rows]);
    }
    Ok(())
}
