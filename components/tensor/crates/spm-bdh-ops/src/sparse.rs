//! The sparse latent: rectify, then gate.

/// Rectifies `src` into `dst`. BDH's only nonlinearity.
pub fn relu_into(src: &[f32], dst: &mut [f32]) {
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot = value.max(0.0);
    }
}

/// `dst = dst * relu(src)`, the gate that makes the latent sparse.
///
/// BDH computes `xy_sparse = x_sparse * y_sparse` with both operands
/// already rectified. Folding the second `relu` into the product saves
/// a pass over `nh * T * N` floats, which at this model's shape is
/// megabytes rather than a micro-optimisation -- see the activation
/// residency discussion in docs/results.md.
///
/// Multiplying in place also lets one buffer serve as both
/// `x_sparse` and `xy_sparse`, halving the largest allocation in the
/// engine.
pub fn scale_product_into(src: &[f32], dst: &mut [f32]) {
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot *= value.max(0.0);
    }
}
