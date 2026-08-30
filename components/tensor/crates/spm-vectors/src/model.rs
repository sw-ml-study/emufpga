//! The golden case value.

use spm_codec::Ternary;
use spm_layout::OpDescriptor;
use spm_numeric::DenseTernary;

/// One reproducible test case.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenCase {
    /// The seed every field was derived from.
    pub seed: u64,
    /// Shape and scale-group size.
    pub descriptor: OpDescriptor,
    /// Weights in stream order (column-major).
    pub weights: Vec<Ternary>,
    /// One scale per group, in stream order.
    pub scales: Vec<f32>,
    /// Activations, one per input column.
    pub activations: Vec<f32>,
}

impl GoldenCase {
    /// The matrix in the shape `spm-numeric`'s reference consumes.
    #[must_use]
    pub fn dense(&self) -> DenseTernary {
        DenseTernary {
            rows: self.descriptor.rows as usize,
            cols: self.descriptor.cols as usize,
            group_size: self.descriptor.group_size as usize,
            weights: self.weights.clone(),
            scales: self.scales.clone(),
        }
    }
}
