//! Golden vectors must be reproducible from a seed and stable on the
//! wire, or the suite quietly rots and stops catching anything.

use spm_numeric::reference_gemv;
use spm_vectors::generate;
use spm_vectors_text::{ParseError, parse, render};

/// Regenerated and compared on every run. If this file and the
/// generator ever disagree, either the generator changed (and every
/// recorded result from before is now unreproducible) or the text
/// format changed (and every committed vector file is stale). Both
/// are deliberate acts that must be made explicitly.
const GOLDEN: &str = include_str!("golden/case-8x4-g8.txt");

#[test]
fn the_generator_reproduces_the_committed_case_exactly() {
    assert_eq!(render(&generate(1, 8, 4, 8)), GOLDEN);
}

#[test]
fn the_same_seed_gives_the_same_case_every_time() {
    // Reproducibility is the property that lets a hardware engineer
    // regenerate a failing case from a single number in a bug report.
    for seed in [1u64, 2, 99, u64::MAX] {
        assert_eq!(generate(seed, 8, 4, 8), generate(seed, 8, 4, 8));
    }
    assert_ne!(generate(1, 8, 4, 8), generate(2, 8, 4, 8));
}

#[test]
fn text_roundtrips_without_loss() {
    // Scales and activations are sixteenths, exactly representable in
    // f32, so a decimal round trip is lossless rather than merely
    // close. That is why the generator draws them that way.
    for (rows, cols, group) in [(8u32, 4u32, 8u32), (3, 2, 4), (16, 5, 16), (1, 1, 64)] {
        let case = generate(7, rows, cols, group);
        let parsed = parse(&render(&case)).expect("parse");
        assert_eq!(parsed, case, "{rows}x{cols} group {group}");
    }
}

#[test]
fn a_parsed_case_computes_the_same_product() {
    // The end that matters: a vector file reloaded from disk must
    // drive the reference to the same answer as the case in memory.
    let case = generate(1, 8, 4, 8);
    let parsed = parse(GOLDEN).expect("parse");
    assert_eq!(
        reference_gemv(&parsed.dense(), &parsed.activations),
        reference_gemv(&case.dense(), &case.activations)
    );
}

#[test]
fn malformed_input_is_rejected_with_the_offending_token() {
    assert_eq!(
        parse("seed 1\nshape 8 4 8\nweights 0+x"),
        Err(ParseError::BadWeight { found: 'x' })
    );
    assert_eq!(
        parse("seed nope"),
        Err(ParseError::BadNumber {
            found: "nope".into()
        })
    );
    assert!(matches!(
        parse("unknown thing"),
        Err(ParseError::BadHeader { .. })
    ));
}
