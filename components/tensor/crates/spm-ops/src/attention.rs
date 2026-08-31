//! Rotary embeddings and softmax attention over resident state.

/// Applies rotary position embedding in place to one head's vector.
///
/// Rotates each adjacent pair by `position * base^(-2i/d)`.
/// TRM's config says `pos_encodings: rope`, and `RoPE` is computed
/// rather than stored -- which is why the checkpoint contains no
/// positional tensors.
///
/// `base` is a parameter rather than a constant because the published
/// value is not in the config we have. 10000 is the near-universal
/// default, and the step that compares against the reference
/// implementation is where it gets confirmed. Until then this is an
/// assumption, and it is written down as one.
///
/// Computed in `f32` throughout, matching what reference `RoPE`
/// implementations do. Position is capped at `u16::MAX`, which is far
/// beyond TRM's 30x30 maze; a model with a long context would need
/// this revisited.
pub fn rope(head: &mut [f32], position: usize, base: f32) {
    let dim = head.len();
    let width = f32::from(u16::try_from(dim).unwrap_or(u16::MAX));
    let at = f32::from(u16::try_from(position).unwrap_or(u16::MAX));
    for pair in 0..dim / 2 {
        let index = f32::from(u16::try_from(pair).unwrap_or(0));
        let theta = base.powf(-2.0 * index / width) * at;
        let (sin, cos) = theta.sin_cos();
        let (x, y) = (head[pair], head[pair + dim / 2]);
        head[pair] = x.mul_add(cos, -(y * sin));
        head[pair + dim / 2] = x.mul_add(sin, y * cos);
    }
}

/// Softmax over `scores`, in place.
fn softmax(scores: &mut [f32]) {
    let Some(max) = scores.iter().copied().reduce(f32::max) else {
        return;
    };
    let mut total = 0.0f32;
    for score in scores.iter_mut() {
        // Shifted by the max before exponentiating: without it a large
        // score overflows to infinity and the whole row becomes NaN.
        // After the shift every term is in (0, 1], so an f32 sum is
        // well conditioned and needs no wider accumulator.
        *score = (*score - max).exp();
        total += *score;
    }
    let scale = 1.0 / total;
    for score in scores.iter_mut() {
        *score *= scale;
    }
}

/// Single-head scaled dot-product attention, resident throughout.
///
/// `q`, `k` and `v` are `[positions][head_dim]` laid out row-major.
/// Writes `[positions][head_dim]` into `out`.
///
/// No causal mask. TRM solves a maze presented at once rather than
/// generating left to right, so every position may attend to every
/// other. A mask here would be wrong, not merely conservative.
///
/// # Panics
/// Panics if the slices do not divide evenly into `head_dim`.
pub fn attend(q: &[f32], k: &[f32], v: &[f32], head_dim: usize, out: &mut [f32]) {
    assert!(head_dim > 0, "head_dim must be positive");
    let positions = q.len() / head_dim;
    let scale = 1.0 / f32::from(u16::try_from(head_dim).unwrap_or(u16::MAX)).sqrt();
    let mut scores = vec![0.0f32; positions];
    for i in 0..positions {
        for (j, score) in scores.iter_mut().enumerate() {
            let dot: f32 = (0..head_dim)
                .map(|d| q[i * head_dim + d] * k[j * head_dim + d])
                .sum();
            *score = dot * scale;
        }
        softmax(&mut scores);
        for d in 0..head_dim {
            out[i * head_dim + d] = (0..positions)
                .map(|j| scores[j] * v[j * head_dim + d])
                .sum();
        }
    }
}

/// Splits a fused qkv projection into heads, applies `RoPE`, attends,
/// and writes the concatenated result.
///
/// `qkv` is `positions x (3 * hidden)` with q, k and v laid out one
/// after another per position -- the layout `gate_up`-style fused
/// projections produce. `shape` is `(heads, head_dim, positions)`.
///
/// # Panics
/// Panics if the buffers do not match `shape`.
pub fn multi_head(qkv: &[f32], shape: (usize, usize, usize), base: f32, out: &mut [f32]) {
    let (count, dim, positions) = shape;
    let width = count * dim;
    let mut q = vec![0.0f32; positions * dim];
    let mut k = vec![0.0f32; positions * dim];
    let mut v = vec![0.0f32; positions * dim];
    let mut head_out = vec![0.0f32; positions * dim];
    for head in 0..count {
        for at in 0..positions {
            let src = at * width * 3 + head * dim;
            let (lo, hi) = (at * dim, (at + 1) * dim);
            q[lo..hi].copy_from_slice(&qkv[src..src + dim]);
            k[lo..hi].copy_from_slice(&qkv[src + width..src + width + dim]);
            v[lo..hi].copy_from_slice(&qkv[src + width * 2..src + width * 2 + dim]);
            rope(&mut q[lo..hi], at, base);
            rope(&mut k[lo..hi], at, base);
        }
        attend(&q, &k, &v, dim, &mut head_out);
        for at in 0..positions {
            out[at * width + head * dim..at * width + (head + 1) * dim]
                .copy_from_slice(&head_out[at * dim..(at + 1) * dim]);
        }
    }
}
