//! The reference must be obviously right, so its tests use values a
//! reader can verify by hand.

use spm_codec::Ternary::{Minus, Plus, Zero};
use spm_numeric::{DenseTernary, cosine_similarity, max_abs_error, mean_error, reference_gemv};

/// A 2x2 matrix in stream order (column-major), one scale per column.
///
///   stream: W[0][0]=+1, W[1][0]=-1, W[0][1]=0, W[1][1]=+1
///   scales: column 0 -> 2.0, column 1 -> 0.5
///   as a matrix:  [ +2   0  ]
///                 [ -2  +0.5]
fn matrix() -> DenseTernary {
    DenseTernary {
        rows: 2,
        cols: 2,
        group_size: 2,
        weights: vec![Plus, Minus, Zero, Plus],
        scales: vec![2.0, 0.5],
    }
}

/// Within tolerance.
///
/// Every value in this file is a small binary fraction and so is
/// exactly representable -- `==` would pass. Comparing floats with
/// `==` is still a habit worth not forming, and clippy's `float_cmp`
/// is right to flag it.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-12
}

#[test]
fn computes_a_hand_checkable_product() {
    // x = [3, 4]
    // y[0] = 2*3 + 0*4    = 6
    // y[1] = -2*3 + 0.5*4 = -4
    let y = reference_gemv(&matrix(), &[3.0, 4.0]);
    assert!(close(y[0], 6.0) && close(y[1], -4.0), "got {y:?}");
}

#[test]
fn error_metrics_agree_on_an_exact_match() {
    let expected = reference_gemv(&matrix(), &[3.0, 4.0]);
    let actual = [6.0f32, -4.0];
    assert!(close(max_abs_error(&actual, &expected), 0.0));
    assert!(close(mean_error(&actual, &expected).expect("mean"), 0.0));
    assert!(close(
        cosine_similarity(&actual, &expected).expect("cosine"),
        1.0
    ));
}

#[test]
fn cosine_catches_a_scale_error_that_direction_survives() {
    // Everything twice as large: max-abs reports a big number without
    // saying why, cosine says the direction is still perfect. That
    // pair is what identifies a mis-applied group scale.
    let expected = reference_gemv(&matrix(), &[3.0, 4.0]);
    let doubled = [12.0f32, -8.0];
    assert!(close(max_abs_error(&doubled, &expected), 6.0));
    assert!(close(
        cosine_similarity(&doubled, &expected).expect("cosine"),
        1.0
    ));
}

#[test]
fn undefined_cases_report_none_rather_than_a_number() {
    assert_eq!(mean_error(&[], &[]), None);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), None);
}
