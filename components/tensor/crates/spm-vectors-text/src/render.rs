//! Rendering a case to text.

use spm_codec::Ternary;
use spm_vectors::GoldenCase;
use std::fmt::Write;

/// Renders a case as text that [`crate::parse`] reads back.
#[must_use]
pub fn render(case: &GoldenCase) -> String {
    let mut out = String::new();
    let d = &case.descriptor;
    let _ = writeln!(out, "seed {}", case.seed);
    let _ = writeln!(out, "shape {} {} {}", d.rows, d.cols, d.group_size);
    let _ = writeln!(out, "scales {}", join(&case.scales));
    let _ = writeln!(out, "activations {}", join(&case.activations));
    for chunk in case.weights.chunks(d.rows.max(1) as usize) {
        let symbols: String = chunk.iter().copied().map(symbol).collect();
        let _ = writeln!(out, "weights {symbols}");
    }
    out
}

/// Space-joins floats.
fn join(values: &[f32]) -> String {
    values
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The one-character symbol for a weight.
fn symbol(weight: Ternary) -> char {
    match weight {
        Ternary::Zero => '0',
        Ternary::Plus => '+',
        Ternary::Minus => '-',
    }
}
