//! RMS norm with a learned scale.

/// `out = x / rms(x) * weight`, per `width` chunk, in place.
///
/// Llama's `RMSNorm` carries a weight vector; TRM's and HRM's do not,
/// which is why [`spm_ops::rms_norm`] takes no scale. Dropping the
/// multiply here would leave the output finite, plausible, and wrong
/// -- the failure mode postmortem defect 7 documents.
///
/// The reciprocal is computed once per row, matching the reference:
/// `hidden * torch.rsqrt(variance + eps)`.
pub fn scaled_rms_norm(values: &mut [f32], weight: &[f32], epsilon: f32, width: usize) {
    for row in values.chunks_mut(width.max(1)) {
        let count = f32::from(u16::try_from(row.len()).unwrap_or(u16::MAX));
        let mean = row.iter().map(|v| v * v).sum::<f32>() / count;
        let scale = 1.0 / (mean + epsilon).sqrt();
        for (slot, gain) in row.iter_mut().zip(weight) {
            *slot = *slot * scale * gain;
        }
    }
}

/// Copies `src` into `dst` and norms it: Llama's pre-norm sublayer input.
///
/// The sublayer reads a normed copy while the residual path keeps the
/// original, which is what makes this pre-norm rather than post-norm.
/// Writing it out at each call site is where that distinction goes
/// quietly wrong.
pub fn pre_norm(src: &[f32], dst: &mut [f32], weight: &[f32], epsilon: f32, width: usize) {
    dst.copy_from_slice(src);
    scaled_rms_norm(dst, weight, epsilon, width);
}

/// `state += delta`, the residual add.
pub fn add_into(state: &mut [f32], delta: &[f32]) {
    for (slot, value) in state.iter_mut().zip(delta) {
        *slot += value;
    }
}
