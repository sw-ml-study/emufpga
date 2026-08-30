//! Assembling a case from a seed.

use crate::draw::{fractions, weights};
use crate::model::GoldenCase;
use spm_layout::{Encoding, OpDescriptor};

/// Builds a case for a `rows` by `cols` matrix with `group_size`
/// weights per scale.
#[must_use]
pub fn generate(seed: u64, rows: u32, cols: u32, group_size: u32) -> GoldenCase {
    let total = rows as usize * cols as usize;
    let groups = total.div_ceil(group_size.max(1) as usize);
    let mut state = seed | 1;
    GoldenCase {
        seed,
        descriptor: OpDescriptor {
            rows,
            cols,
            group_size,
            encoding: Encoding::Ternary2F32I32,
            lane_count: 1,
        },
        weights: weights(&mut state, total),
        scales: fractions(&mut state, groups, 32, 0.0625),
        activations: fractions(&mut state, cols as usize, 64, -2.0),
    }
}
