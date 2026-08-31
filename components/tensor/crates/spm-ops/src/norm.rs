//! Root-mean-square normalization.

/// Normalizes `values` in place by their RMS.
///
/// No learned gain. TRM calls `rms_norm(x, variance_epsilon)` with no
/// weight parameter, which is why its checkpoint holds no norm
/// tensors -- inventing one here would produce a model that silently
/// disagreed with the published one.
///
/// `epsilon` is added to the mean square before the square root, as
/// `1e-5` in TRM's config.
pub fn rms_norm(values: &mut [f32], epsilon: f32) {
    if values.is_empty() {
        return;
    }
    // Accumulated in f32. The terms are squares of activations, all
    // the same sign and similar magnitude, so the sum is well
    // conditioned across a row of a few thousand; a wider accumulator
    // would buy nothing and would force a narrowing cast back.
    let sum: f32 = values.iter().map(|v| v * v).sum();
    let count = f32::from(u16::try_from(values.len()).unwrap_or(u16::MAX));
    let scale = 1.0 / (sum / count + epsilon).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// TRM's post-norm residual: `state = rms_norm(state + delta)`.
///
/// Post-norm rather than pre-norm, matching `trm.py`, which normalizes
/// *after* adding the sublayer output rather than before feeding it.
/// The two are not interchangeable and produce different models.
pub fn residual_norm(state: &mut [f32], delta: &[f32], epsilon: f32) {
    for (slot, add) in state.iter_mut().zip(delta) {
        *slot += add;
    }
    rms_norm(state, epsilon);
}
