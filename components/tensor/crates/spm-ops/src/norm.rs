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

/// TRM's post-norm residual: `state = rms_norm(state + delta)`, with
/// the norm taken **per position**.
///
/// Post-norm rather than pre-norm, matching `trm.py`, which normalizes
/// after adding the sublayer output rather than before feeding it. The
/// two are not interchangeable and produce different models.
///
/// `width` is the model width, and passing it is not optional
/// bookkeeping. The reference computes
/// `hidden_states.pow(2).mean(-1, keepdim=True)` -- a mean over the
/// LAST axis, so every position is normalized by its own RMS. An
/// earlier version here normalized the whole state as a single vector,
/// which is wrong in a way that is easy to miss: with a handful of
/// positions the scales are similar, the output stays finite and
/// plausible, and only a numerical comparison against the reference
/// shows it. It cost a cosine of 0.9993 where the stages either side
/// of it were exact.
pub fn residual_norm(state: &mut [f32], delta: &[f32], epsilon: f32, width: usize) {
    for (slot, add) in state.iter_mut().zip(delta) {
        *slot += add;
    }
    for row in state.chunks_mut(width.max(1)) {
        rms_norm(row, epsilon);
    }
}

/// `LayerNorm` with no affine parameters, over each `width` chunk.
///
/// Distinct from [`rms_norm`], and the difference is the mean. RMS
/// norm divides by the root-mean-square and leaves the centre where it
/// is; this subtracts the mean first. TRM and HRM use the former, BDH
/// uses the latter -- `nn.LayerNorm(D, elementwise_affine=False)` --
/// and substituting one for the other is a silent accuracy bug of
/// exactly the kind postmortem defect 7 was.
///
/// The variance is biased (divided by `width`, not `width - 1`),
/// matching torch.
pub fn layer_norm(values: &mut [f32], epsilon: f32, width: usize) {
    for row in values.chunks_mut(width.max(1)) {
        let count = f32::from(u16::try_from(row.len()).unwrap_or(u16::MAX));
        let mean = row.iter().sum::<f32>() / count;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / count;
        let scale = 1.0 / (variance + epsilon).sqrt();
        for slot in row.iter_mut() {
            *slot = (*slot - mean) * scale;
        }
    }
}

/// `state = layer_norm(state + delta)`, per `width` chunk.
///
/// The [`layer_norm`] counterpart to [`residual_norm`], and it exists
/// for the same reason: a post-norm residual is one operation in every
/// architecture that uses it, and writing it out at each call site is
/// where an axis or an ordering quietly goes wrong.
pub fn residual_layer_norm(state: &mut [f32], delta: &[f32], epsilon: f32, width: usize) {
    for (slot, value) in state.iter_mut().zip(delta) {
        *slot += value;
    }
    layer_norm(state, epsilon, width);
}
