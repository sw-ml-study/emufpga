//! Grouped-query attention with a causal mask.

use core::iter::zip;
use spm_ops::rope;

/// Rotates every head of a projection in place, at its own position.
///
/// `buf` is `positions x (heads * head_dim)`. Applied to queries and
/// keys but never to values, which is what the reference does.
pub fn rotate_heads(buf: &mut [f32], shape: (usize, usize, usize), base: f32) {
    let (positions, heads, head_dim) = shape;
    for position in 0..positions {
        let row = position * heads * head_dim;
        for head in 0..heads {
            let at = row + head * head_dim;
            rope(&mut buf[at..at + head_dim], position, base);
        }
    }
}

/// Softmax over a causal prefix, in place.
///
/// Shifted by the max before exponentiating: without it a large score
/// overflows to infinity and the row becomes NaN.
fn softmax(scores: &mut [f32]) {
    let Some(max) = scores.iter().copied().reduce(f32::max) else {
        return;
    };
    let mut total = 0.0f32;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        total += *score;
    }
    for score in scores.iter_mut() {
        *score /= total;
    }
}

/// Attends one query position over keys `0..=position`, one head.
///
/// The mask **includes** the diagonal: a token attends to itself.
/// BDH's excludes it. Getting this wrong shifts every output by one
/// position and still produces finite numbers.
///
/// `scores` is the caller's buffer, reused across every head and
/// position. Allocating here would mean one allocation per head per
/// position per layer -- 30 * 9 * T of them per forward.
fn attend_position(
    query: &[f32],
    kv: (&[f32], &[f32]),
    at: (usize, usize, usize),
    buffers: (&mut Vec<f32>, &mut [f32]),
) {
    let ((keys, values), (position, offset, stride)) = (kv, at);
    let (scores, out) = buffers;
    let width = query.len();
    let scale = 1.0 / f32::from(u16::try_from(width).unwrap_or(1)).sqrt();
    scores.clear();
    scores.extend((0..=position).map(|key| {
        let at = key * stride + offset;
        let dot: f32 = zip(query, &keys[at..at + width]).map(|(a, b)| a * b).sum();
        dot * scale
    }));
    softmax(scores);
    out.fill(0.0);
    for (key, weight) in scores.iter().enumerate() {
        let at = key * stride + offset;
        for (slot, value) in out.iter_mut().zip(&values[at..at + width]) {
            *slot = weight.mul_add(*value, *slot);
        }
    }
}

/// Grouped-query causal attention.
///
/// `queries` is `positions x (heads * head_dim)`; `keys` and `values`
/// are `positions x (kv_heads * head_dim)`. Query head `h` reads KV
/// head `h / (heads / kv_heads)` -- 9 heads over 3 KV heads means
/// three consecutive query heads share one key and value.
///
/// # Panics
/// Panics if `kv_heads` is zero or does not divide `heads`, which is
/// a config error rather than an input error.
pub fn grouped_causal(
    queries: &[f32],
    kv: (&[f32], &[f32]),
    shape: (usize, usize, usize, usize),
    out: &mut [f32],
) {
    let (positions, heads, kv_heads, head_dim) = shape;
    assert!(
        kv_heads > 0 && heads % kv_heads == 0,
        "heads must group evenly"
    );
    let (group, stride) = (heads / kv_heads, kv_heads * head_dim);
    let mut scores = Vec::with_capacity(positions);
    let mut scratch = vec![0.0f32; head_dim];
    for position in 0..positions {
        for head in 0..heads {
            let at = position * heads * head_dim + head * head_dim;
            let place = (position, (head / group) * head_dim, stride);
            let buffers = (&mut scores, &mut scratch[..]);
            attend_position(&queries[at..at + head_dim], kv, place, buffers);
            out[at..at + head_dim].copy_from_slice(&scratch);
        }
    }
}
