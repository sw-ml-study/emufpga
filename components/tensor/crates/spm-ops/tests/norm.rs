//! `layer_norm` against hand-computed values, and against `rms_norm`.
//!
//! The two are one subtraction apart and substituting either for the
//! other is a silent accuracy bug -- the same shape as postmortem
//! defect 7, which cost an afternoon. TRM and HRM use `rms_norm`; BDH
//! uses `layer_norm`. Nothing in a shape check can tell them apart.

use spm_ops::{layer_norm, rms_norm};

#[test]
fn layer_norm_centres_and_scales_each_row() {
    // [1,2,3,4]: mean 2.5, biased variance 1.25, sd ~1.118.
    let mut values = vec![1.0f32, 2.0, 3.0, 4.0];
    layer_norm(&mut values, 1e-5, 4);
    let expected = [-1.341_64, -0.447_21, 0.447_21, 1.341_64];
    for (got, want) in values.iter().zip(expected) {
        assert!(
            (got - want).abs() < 1e-4,
            "got {values:?}, want {expected:?}"
        );
    }
    let mean: f32 = values.iter().sum::<f32>() / 4.0;
    assert!(
        mean.abs() < 1e-5,
        "centred row should have zero mean, got {mean}"
    );
}

#[test]
fn layer_norm_normalizes_rows_independently() {
    // The defect-7 property, for the other norm: two rows differing
    // only by an offset and a scale must come out identical.
    let mut values = vec![1.0f32, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0];
    layer_norm(&mut values, 1e-5, 4);
    for i in 0..4 {
        let (a, b) = (values[i], values[4 + i]);
        assert!((a - b).abs() < 1e-4, "position {i}: {a} vs {b}");
    }
}

#[test]
fn layer_norm_is_not_rms_norm() {
    // Guards the substitution directly. On a row with a nonzero mean
    // the two disagree, and if this ever stops being true one of them
    // has been quietly redefined.
    let row = vec![1.0f32, 2.0, 3.0, 4.0];
    let (mut centred, mut scaled) = (row.clone(), row);
    layer_norm(&mut centred, 1e-5, 4);
    rms_norm(&mut scaled, 1e-5);
    let gap = centred
        .iter()
        .zip(&scaled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        gap > 0.5,
        "layer_norm and rms_norm should differ, gap {gap}"
    );
    assert!(
        scaled.iter().sum::<f32>() > 1.0,
        "rms_norm must NOT centre; if it does, it is layer_norm"
    );
}
