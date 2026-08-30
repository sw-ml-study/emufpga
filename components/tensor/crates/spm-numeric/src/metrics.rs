//! How far apart the engine's output is from the f64 reference.
//!
//! `actual` is what the streaming engine produced (`f32`); `expected`
//! is the reference (`f64`). Widening `f32` to `f64` is lossless, so
//! every comparison happens at the oracle's precision.

/// Largest absolute difference between corresponding elements.
///
/// The strictest of the three, and the one a regression should be
/// judged on: an average can hide a single catastrophically wrong
/// output, which for a matrix-vector product means one broken row.
#[must_use]
pub fn max_abs_error(actual: &[f32], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (f64::from(*a) - e).abs())
        .fold(0.0f64, f64::max)
}

/// Mean absolute difference.
///
/// Returns `None` for empty inputs rather than a misleading zero.
#[must_use]
pub fn mean_error(actual: &[f32], expected: &[f64]) -> Option<f64> {
    let count = actual.len().min(expected.len());
    (count > 0).then(|| {
        let total: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (f64::from(*a) - e).abs())
            .sum();
        total / f64::from(u32::try_from(count).unwrap_or(u32::MAX))
    })
}

/// Euclidean norm of a slice already widened to `f64`.
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}

/// Cosine similarity, `1.0` for identical directions.
///
/// Insensitive to overall magnitude, so it catches "right shape,
/// wrong scale" -- exactly the failure a mis-applied group scale
/// produces, and exactly the failure `max_abs_error` reports as a
/// large number without saying why. Returns `None` if either vector
/// is all zeros, where the angle is undefined.
///
/// Clamped to `[-1, 1]`. Cosine is defined on that interval, but
/// summing the dot product and the norms separately in floating point
/// can land just outside it -- an exact match measured 1.0000000000000002
/// here. Returning that would be reporting rounding noise as if it
/// were a similarity above perfect, and would break any caller that
/// reasonably asserts the result is at most 1.
#[must_use]
pub fn cosine_similarity(actual: &[f32], expected: &[f64]) -> Option<f64> {
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| f64::from(*a) * e)
        .sum();
    let scale = norm(actual.iter().map(|v| f64::from(*v))) * norm(expected.iter().copied());
    (scale > 0.0).then(|| (dot / scale).clamp(-1.0, 1.0))
}
