//! Guards for defects recorded in docs/postmortem-1.md.
//!
//! Each test names the defect it prevents. They exist because the
//! original bug was found by a manual reference comparison that needs
//! torch and hundreds of megabytes of weights -- neither of which can
//! run in a gate. These reproduce the *property* the reference
//! checked, hermetically, so a regression fails `just check` instead
//! of waiting for someone to re-run a comparison by hand.

use spm_ops::{residual_norm, rms_norm};

/// Positions per row in these fixtures. Paired with its own f32 form
/// rather than cast at the point of use: `usize as f32` is a
/// precision-loss lint, and the width here is a constant the test
/// chose, not a value it received.
const WIDTH: usize = 4;
const WIDTH_F32: f32 = 4.0;

#[test]
fn defect_7_rms_norm_normalizes_each_position_independently() {
    // POSTMORTEM DEFECT 7. residual_norm normalized the whole state as
    // one vector; the reference takes mean(-1), so every position is
    // normalized by its own RMS.
    //
    // The bug is invisible to a "finite and changed" assertion, which
    // is what the tests had. It is obvious to this one: two rows that
    // differ only by a scale factor must come out IDENTICAL, because
    // each is divided by its own RMS. Under a global norm the larger
    // row stays larger.
    let width = WIDTH;
    let mut state = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let zero = vec![0.0f32; state.len()];
    residual_norm(&mut state, &zero, 1e-5, width);

    for i in 0..width {
        let (small, large) = (state[i], state[width + i]);
        assert!(
            (small - large).abs() < 1e-4,
            "position {i}: {small} vs {large} -- rows differing only by scale \
             must normalize to the same values, or the norm is global"
        );
    }
    // And each row really is unit-RMS, not merely equal to the other.
    let rms: f32 = (state[..width].iter().map(|v| v * v).sum::<f32>() / WIDTH_F32).sqrt();
    assert!((rms - 1.0).abs() < 1e-3, "row RMS {rms} should be ~1");
}

#[test]
fn defect_7_a_global_norm_would_fail_the_test_above() {
    // Demonstrates the guard actually discriminates. Normalizing the
    // whole slice at once -- the old, wrong behaviour -- leaves the
    // two rows an order of magnitude apart.
    let width = WIDTH;
    let mut whole = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    rms_norm(&mut whole, 1e-5);
    let ratio = whole[width] / whole[0];
    assert!(
        ratio > 5.0,
        "a global norm must leave the rows unequal (ratio {ratio}); \
         if this ever passes, the discriminating power of defect_7 is gone"
    );
}
