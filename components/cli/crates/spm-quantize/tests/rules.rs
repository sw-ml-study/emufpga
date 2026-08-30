//! The quantization rule, checked against values worked by hand.
//!
//! This transform is lossy and the loss is the point of the
//! architecture, so every rule it applies is pinned here rather than
//! left to whatever the code happens to do.

use spm_codec::Ternary::{Minus, Plus, Zero};
use spm_quantize::{parse_matrix, quantize};

#[test]
fn absmean_scale_and_rounding_are_what_the_docs_claim() {
    // One group covering all four weights: |w| = 1, 3, 0, 4.
    // mean = 8/4 = 2.0.
    //   1/2 = 0.5  -> round 1  (ties away from zero) -> Plus
    //  -3/2 = -1.5 -> round -2 -> clamped            -> Minus
    //   0/2 = 0                                      -> Zero
    //   4/2 = 2    -> clamped                        -> Plus
    let matrix = parse_matrix("1 -3\n0 4\n").expect("parse");
    // Stream order is column-major: 1, 0, -3, 4.
    let q = quantize(&matrix, 4);
    assert!((q.scales[0] - 2.0).abs() < 1e-6, "scale {:?}", q.scales);
    assert_eq!(q.weights, vec![Plus, Zero, Minus, Plus]);
}

#[test]
fn a_half_scale_weight_rounds_away_from_zero() {
    // Ties matter and f32::round breaks them away from zero, so a
    // weight at exactly half its group scale becomes +/-1, not 0.
    let matrix = parse_matrix("1 1 1 -1\n").expect("parse");
    let q = quantize(&matrix, 4);
    assert!((q.scales[0] - 1.0).abs() < 1e-6);
    assert_eq!(q.weights, vec![Plus, Plus, Plus, Minus]);
}

#[test]
fn an_all_zero_group_gets_scale_one_not_scale_zero() {
    // A zero scale would be arithmetically fine but would put a value
    // into the wire format that the engine multiplies an activation by
    // on every column. 1.0 dequantizes back to zero exactly.
    let matrix = parse_matrix("0 0\n0 0\n").expect("parse");
    let q = quantize(&matrix, 4);
    assert!((q.scales[0] - 1.0).abs() < 1e-6, "scale {:?}", q.scales);
    assert_eq!(q.weights, vec![Zero; 4]);
}

#[test]
fn each_group_gets_its_own_scale() {
    // A loud column must not flatten a quiet one. Column 0 has
    // absmean 1, column 1 has absmean 100; with group_size == rows
    // each column is its own group and both keep full resolution.
    let matrix = parse_matrix("1 100\n-1 -100\n").expect("parse");
    let q = quantize(&matrix, 2);
    assert_eq!(q.scales.len(), 2);
    assert!((q.scales[0] - 1.0).abs() < 1e-6);
    assert!((q.scales[1] - 100.0).abs() < 1e-4);
    assert_eq!(q.weights, vec![Plus, Minus, Plus, Minus]);
}
