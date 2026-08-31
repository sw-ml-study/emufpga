//! The streamed engine must reproduce the naive f64 reference.

mod support;

use spm_gemv_ref::{Activations, run_gemv};
use spm_numeric::{cosine_similarity, max_abs_error, reference_gemv};
use spm_stream_mem::MemoryWeightStream;
use spm_vectors::generate;
use support::to_spm;

/// Tolerance for a single output element.
///
/// Where the number comes from: the engine accumulates in `f32` while
/// the reference accumulates in `f64`, so the gap is `f32` summation
/// error over at most `cols` terms. With activations bounded by 2,
/// scales by 2, and `cols <= 64`, the worst-case magnitude of any
/// output is ~256, and `f32` has ~1.2e-7 relative precision, giving
/// roughly 256 * 64 * 1.2e-7 ~= 2e-3 as a generous bound. Set once
/// from that estimate and NOT tuned upward to make a test pass -- if
/// a case exceeds it, the engine is wrong, not the tolerance.
const TOLERANCE: f64 = 2e-3;

#[test]
fn matches_the_reference_across_shapes_and_group_sizes() {
    // Group sizes chosen to cover: smaller than a column, exactly a
    // column (the recommended configuration), spanning columns, and
    // not dividing the matrix evenly.
    let cases = [
        (3u32, 2u32, 4u32),
        (8, 4, 8),
        (8, 4, 3),
        (16, 5, 16),
        (7, 3, 64),
        (1, 1, 64),
    ];
    for (index, (rows, cols, group)) in cases.into_iter().enumerate() {
        let case = generate(index as u64 + 1, rows, cols, group);
        let expected = reference_gemv(&case.dense(), &case.activations);

        let stream = MemoryWeightStream::new(to_spm(&case));
        let outcome =
            run_gemv(stream, &Activations::broadcast(1, &case.activations)).expect("run gemv");

        let actual = outcome.bank.lane(0);
        let error = max_abs_error(&actual, &expected);
        assert!(
            error <= TOLERANCE,
            "{rows}x{cols} group {group}: max abs error {error} exceeds {TOLERANCE}\n\
             actual   {actual:?}\nexpected {expected:?}"
        );
        // Cosine catches "right shape, wrong scale", which max-abs
        // reports only as a large number without saying why.
        //
        // It returns None when either vector is all zeros, where the
        // angle is undefined -- a generated case whose only weight is
        // Zero does exactly that. Skipping is correct rather than
        // lenient: max_abs_error above already covers those cases
        // completely, since it compares against zero directly.
        if let Some(similarity) = cosine_similarity(&actual, &expected) {
            assert!(
                similarity > 0.999_999,
                "{rows}x{cols} group {group}: direction diverged ({similarity})"
            );
        }
    }
}

#[test]
fn a_single_element_matrix_is_handled() {
    let case = generate(7, 1, 1, 64);
    let expected = reference_gemv(&case.dense(), &case.activations);
    let outcome = run_gemv(
        MemoryWeightStream::new(to_spm(&case)),
        &Activations::broadcast(1, &case.activations),
    )
    .expect("run gemv");
    assert!(max_abs_error(&outcome.bank.lane(0), &expected) <= TOLERANCE);
}

#[test]
fn a_zero_weight_matrix_accumulates_nothing() {
    let case = generate(11, 0, 4, 8);
    let outcome = run_gemv(
        MemoryWeightStream::new(to_spm(&case)),
        &Activations::broadcast(1, &case.activations),
    )
    .expect("run gemv");
    assert_eq!(outcome.bank.rows, 0);
    assert_eq!(outcome.metrics.weights_decoded, 0);
}

#[test]
fn too_few_activations_fail_before_the_scan_starts() {
    let case = generate(13, 8, 4, 8);
    let error = run_gemv(
        MemoryWeightStream::new(to_spm(&case)),
        &Activations::broadcast(1, &case.activations[..2]),
    )
    .expect_err("must fail");
    assert!(
        format!("{error}").contains("need 4 activations"),
        "got {error}"
    );
}
