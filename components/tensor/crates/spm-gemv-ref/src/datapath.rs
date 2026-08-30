//! The datapath: what happens to each weight as it arrives.
//!
//! Kept in its own file so whoever writes the RTL has one place to
//! read. Everything here maps to wires.

use spm_accum::AccumulatorBank;
use spm_activations::Activations;
use spm_codec::{NEGATIVE_BIT, NONZERO_BIT, code_at};

/// Applies one scale group's weights to the accumulators.
///
/// This function owns the stream *geometry*: which column a stream
/// position falls in, and when to refresh the scaled activations. The
/// arithmetic itself is [`apply_weight`].
///
/// The loop is bounded by `count`, never by the packed length. A short
/// final group leaves padding bits in its last byte, and iterating
/// those would feed phantom zero weights into the scan.
pub(crate) fn apply_group(
    bank: &mut AccumulatorBank,
    activations: &Activations,
    scaled: &mut [f32],
    group: (f32, &[u8], usize),
    at: (usize, usize),
) {
    let (scale, packed, count) = group;
    let (position, rows) = at;
    let mut current_col = usize::MAX;
    for local in 0..count {
        let Some(code) = code_at(packed, local) else {
            break;
        };
        let index = position + local;
        let col = index / rows;
        if col != current_col {
            activations.scale_column(scale, col, scaled);
            current_col = col;
        }
        apply_weight(bank, code, index % rows, scaled);
    }
}

/// One weight, one cycle. The whole arithmetic unit.
///
/// `NONZERO_BIT` is the accumulator enable and `NEGATIVE_BIT` is the
/// add/subtract select. They are read as masks rather than decoded
/// into [`spm_codec::Ternary`] because in the fabric they are not a
/// value at all -- they are two wires arriving off the stream. There
/// is no multiplier here, and there is deliberately nowhere to put
/// one.
fn apply_weight(bank: &mut AccumulatorBank, code: u8, row: usize, scaled: &[f32]) {
    if code & NONZERO_BIT != 0 {
        bank.accumulate(row, code & NEGATIVE_BIT != 0, scaled);
    }
}
