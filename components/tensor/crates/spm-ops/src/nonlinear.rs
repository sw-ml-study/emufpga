//! `SwiGLU` and the sigmoid-linear unit behind it.

/// Sigmoid-linear unit: `x * sigmoid(x)`, also called swish.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `SwiGLU`: `silu(gate) * up`, written into `out`.
///
/// TRM's `MLP` projects to twice the intermediate width in one matrix
/// (`gate_up_proj`) and splits the result. Keeping the split here
/// rather than in the model means the streamed matmul stays a plain
/// `y = Wx` and never has to know what the halves mean.
///
/// # Panics
/// Panics if `gate_up` is not exactly twice `out`, which would mean
/// the projection and the model disagree about the intermediate size.
pub fn swiglu(gate_up: &[f32], out: &mut [f32]) {
    assert_eq!(
        gate_up.len(),
        out.len() * 2,
        "gate_up must be twice the intermediate width"
    );
    let (gate, up) = gate_up.split_at(out.len());
    for ((slot, g), u) in out.iter_mut().zip(gate).zip(up) {
        *slot = silu(*g) * u;
    }
}

/// `SwiGLU` across a batch of positions.
///
/// `gate_up` is `positions x (2 * inter)`, `out` is
/// `positions x inter`. Batched here rather than in the model so the
/// model reads as four streamed projections and nothing else.
///
/// # Panics
/// Panics if the buffers do not match `positions` and `inter`.
pub fn swiglu_batch(gate_up: &[f32], out: &mut [f32], inter: usize, positions: usize) {
    for at in 0..positions {
        let wide = at * inter * 2..(at + 1) * inter * 2;
        let narrow = at * inter..(at + 1) * inter;
        swiglu(&gate_up[wide], &mut out[narrow]);
    }
}
