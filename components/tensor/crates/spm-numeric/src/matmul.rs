//! A deliberately naive dense reference for `y = Wx`.

use spm_codec::Ternary;

/// A ternary matrix in **stream order**, with one scale per group.
///
/// Stream order is column-major (docs/spm-format.md): position `k`
/// holds `W[k % rows][k / rows]`. Holding it this way rather than as
/// rows of a matrix keeps the reference honest -- it consumes exactly
/// the ordering the format defines, so a tiling bug cannot hide behind
/// a convenient in-memory layout.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseTernary {
    /// Output count, M.
    pub rows: usize,
    /// Input count, N.
    pub cols: usize,
    /// Weights per scale group, in stream order.
    pub group_size: usize,
    /// Weights in stream order, `rows * cols` of them.
    pub weights: Vec<Ternary>,
    /// One scale per group, in stream order.
    pub scales: Vec<f32>,
}

/// Computes `y = Wx` the obvious way, in `f64`.
///
/// Returns `f64` rather than narrowing to `f32`. The oracle should
/// carry more precision than the thing it judges, not the same amount
/// -- narrowing here would fold the reference's own rounding into
/// every tolerance and make a real engine error indistinguishable
/// from float luck.
///
/// # Panics
/// Panics if `activations` is shorter than `cols`, or if the matrix
/// is internally inconsistent. Both are test-setup bugs.
#[must_use]
pub fn reference_gemv(matrix: &DenseTernary, activations: &[f32]) -> Vec<f64> {
    assert!(activations.len() >= matrix.cols, "too few activations");
    let mut out = vec![0.0f64; matrix.rows];
    for (index, weight) in matrix.weights.iter().enumerate() {
        let scale = matrix.scales[index / matrix.group_size];
        let row = index % matrix.rows;
        let col = index / matrix.rows;
        out[row] += f64::from(weight.value()) * f64::from(scale) * f64::from(activations[col]);
    }
    out
}
