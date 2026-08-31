//! Linear attention over the sparse latent, with no learned weights.

use core::f32::consts::TAU;

/// BDH's rotary frequencies for a latent of `n` entries.
///
/// `1 / theta^(quantize(i) / n) / 2pi`, where `quantize` floors to
/// even indices so a rotation pair shares one frequency. The division
/// by `2pi` is in the reference and matters: phases are reduced modulo
/// 1 before being scaled back up by `2pi`, so the frequency is in
/// turns, not radians.
#[must_use]
pub fn freqs(n: usize, theta: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let quantized = f32::from(u16::try_from(i / 2 * 2).unwrap_or(u16::MAX));
            let width = f32::from(u16::try_from(n).unwrap_or(u16::MAX));
            1.0 / theta.powf(quantized / width) / TAU
        })
        .collect()
}

/// Rotates one row in place at `position`, pairwise.
///
/// `(v0, v1) -> (v0*cos - v1*sin, v1*cos + v0*sin)`, which is the
/// reference's `stack((-v[1::2], v[::2]))` written out. Phases are
/// taken modulo 1 first, exactly as `phases_cos_sin` does -- for
/// `theta = 65536` and long sequences the raw product is large enough
/// that skipping the reduction loses precision.
pub fn rotate(row: &mut [f32], position: usize, freqs: &[f32]) {
    let turns = f32::from(u16::try_from(position).unwrap_or(u16::MAX));
    for (pair, frequency) in row.chunks_mut(2).zip(freqs.iter().step_by(2)) {
        if pair.len() < 2 {
            break;
        }
        let phase = (turns * frequency).rem_euclid(1.0) * TAU;
        let (cos, sin) = (phase.cos(), phase.sin());
        let (even, odd) = (pair[0], pair[1]);
        pair[0] = even.mul_add(cos, -(odd * sin));
        pair[1] = odd.mul_add(cos, even * sin);
    }
}

/// `out = tril(Q Q^T, -1) V`, for one head.
///
/// `queries` is `positions x latent`, already rotated. `values` is
/// `positions x width`. The mask is **strictly** lower triangular:
/// `.tril(diagonal=-1)` in the reference, so position `t` attends to
/// `0..t` and never to itself. Position 0 therefore attends to
/// nothing and contributes a zero row, which is correct rather than a
/// degenerate case to guard against.
///
/// There are no scores to keep: each row's weights are consumed into
/// `out` as they are produced, so the `T x T` matrix the reference
/// materialises is never allocated here.
pub fn scores_times_values(
    queries: &[f32],
    values: &[f32],
    shape: (usize, usize, usize),
    out: &mut [f32],
) {
    let (positions, latent, width) = shape;
    out.fill(0.0);
    for query in 0..positions {
        let row = &queries[query * latent..(query + 1) * latent];
        for key in 0..query {
            let other = &queries[key * latent..(key + 1) * latent];
            let score: f32 = row.iter().zip(other).map(|(a, b)| a * b).sum();
            let source = &values[key * width..(key + 1) * width];
            let target = &mut out[query * width..(query + 1) * width];
            for (slot, value) in target.iter_mut().zip(source) {
                *slot = score.mul_add(*value, *slot);
            }
        }
    }
}

/// Runs BDH's attention for every head.
///
/// `sparse` is positions-major, `positions x (heads * latent)`, the
/// layout the decoder sweep wants. `out` is **heads-major**,
/// `heads x positions x width`, so each head's result is contiguous
/// and can be fed straight to the next streamed matmul without a
/// gather.
///
/// `scratch` holds one head's rotated queries. The rotation cannot be
/// done in place: `x_sparse` is needed unrotated for the gate that
/// follows, and rotating it there would be a bug no shape check could
/// catch.
pub fn attend_heads(
    sparse: &[f32],
    values: &[f32],
    shape: (usize, usize, usize, usize),
    freqs: &[f32],
    buffers: (&mut [f32], &mut [f32]),
) {
    let (positions, latent, width, heads) = shape;
    let (scratch, out) = buffers;
    let stride = heads * latent;
    for head in 0..heads {
        for position in 0..positions {
            let at = position * stride + head * latent;
            let row = &mut scratch[position * latent..(position + 1) * latent];
            row.copy_from_slice(&sparse[at..at + latent]);
            rotate(row, position, freqs);
        }
        let span = positions * width;
        let target = &mut out[head * span..(head + 1) * span];
        scores_times_values(scratch, values, (positions, latent, width), target);
    }
}
