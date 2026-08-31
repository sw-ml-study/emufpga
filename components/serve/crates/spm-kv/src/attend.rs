//! One query position against a cached prefix.

use core::iter::zip;
use spm_ops::rope;

/// Rotates one token's heads at its **absolute** position.
///
/// Decoding is one token at a time, so a position-from-row-index
/// helper like `spm_smol_ops::rotate_heads` would rotate every client
/// at position 0. Each client is at a different point in its own
/// sequence, and rotating at the wrong position produces fluent
/// nonsense rather than an error.
pub fn rotate_at(buf: &mut [f32], shape: (usize, usize), position: usize, base: f32) {
    let (heads, head_dim) = shape;
    for head in 0..heads {
        let at = head * head_dim;
        rope(&mut buf[at..at + head_dim], position, base);
    }
}

/// Softmax over a prefix, in place. Shifted by the max first.
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

/// One head, one query position, over the cached prefix.
fn attend_head(
    query: &[f32],
    cached: (&[f32], &[f32]),
    at: (usize, usize, usize),
    buffers: (&mut Vec<f32>, &mut [f32]),
) {
    let (scores, out) = buffers;
    let ((keys, values), (positions, offset, stride), width) = (cached, at, query.len());
    let scale = 1.0 / f32::from(u16::try_from(width).unwrap_or(1)).sqrt();
    let dot = |p: usize| {
        let key = &keys[p * stride + offset..][..width];
        zip(query, key).map(|(a, b)| a * b).sum::<f32>() * scale
    };
    scores.clear();
    scores.extend((0..positions).map(dot));
    softmax(scores);
    out.fill(0.0);
    for (p, weight) in scores.iter().enumerate() {
        let v = p * stride + offset;
        for (slot, value) in zip(out.iter_mut(), &values[v..v + width]) {
            *slot = weight.mul_add(*value, *slot);
        }
    }
}

/// Grouped-query attention for ONE position over a cached prefix.
///
/// `query` is `heads * head_dim` for the single token being decoded.
/// `keys` and `values` come from the client's cache and already
/// include this token's own key and value -- the mask includes the
/// diagonal, so a token attends to itself.
///
/// This is the work that does **not** amortize. It carries no weights
/// and is entirely per-client, which is exactly why it can live in
/// ordinary memory while the parameters stream past.
///
/// # Panics
/// Panics if `kv_heads` does not divide `heads`.
pub fn attend_cached(
    query: &[f32],
    cached: (&[f32], &[f32]),
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
    for head in 0..heads {
        let at = head * head_dim;
        let place = (positions, (head / group) * head_dim, stride);
        let (row, target) = (&query[at..at + head_dim], &mut out[at..at + head_dim]);
        attend_head(row, cached, place, (&mut scores, target));
    }
}
