//! Parsing text back into a case.

use crate::fields::{self, ParseError};
use spm_codec::Ternary;
use spm_layout::{Encoding, OpDescriptor};
use spm_vectors::GoldenCase;

/// The fields a case accumulates while its lines are read.
#[derive(Default)]
struct Partial {
    seed: u64,
    shape: [u32; 3],
    scales: Vec<f32>,
    activations: Vec<f32>,
    weights: Vec<Ternary>,
}

/// Parses text produced by [`crate::render`].
///
/// # Errors
/// Returns [`ParseError`] if a header line is missing or malformed, a
/// weight character is unrecognised, or a number does not parse.
pub fn parse(text: &str) -> Result<GoldenCase, ParseError> {
    let mut partial = Partial::default();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        absorb(&mut partial, line)?;
    }
    Ok(GoldenCase {
        seed: partial.seed,
        descriptor: OpDescriptor {
            rows: partial.shape[0],
            cols: partial.shape[1],
            group_size: partial.shape[2],
            encoding: Encoding::Ternary2F32I32,
            lane_count: 1,
        },
        weights: partial.weights,
        scales: partial.scales,
        activations: partial.activations,
    })
}

/// Folds one line into the case being built.
fn absorb(partial: &mut Partial, line: &str) -> Result<(), ParseError> {
    let (tag, rest) = line.split_once(' ').unwrap_or((line, ""));
    match tag {
        "seed" => {
            partial.seed = rest.trim().parse().map_err(|_| ParseError::BadNumber {
                found: rest.trim().into(),
            })?;
        }
        "shape" => partial.shape = fields::shape(rest)?,
        "scales" => partial.scales = fields::numbers(rest)?,
        "activations" => partial.activations = fields::numbers(rest)?,
        "weights" => partial.weights.extend(fields::weights(rest)?),
        _ => return Err(ParseError::BadHeader { line: line.into() }),
    }
    Ok(())
}
