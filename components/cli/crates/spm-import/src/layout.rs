//! The host-side layout: what order the streams are in, and what
//! they are called.
//!
//! Both are host concerns. The FPGA streams bytes in the order the
//! directory declares and never needs a name; what it does need is
//! for that order to be the order it consumes them in, which is what
//! the order file specifies.

use crate::manifest::Tensor;
use std::fmt::Write;

/// Renders the stream-index-to-name mapping.
///
/// Text, tab separated, one line per stream, in directory order. Text
/// because the reader is a human debugging a mismatch, or a scheduler
/// in whatever language; a binary table would buy nothing at this size
/// and cost everyone a parser.
#[must_use]
pub fn render_sidecar(tensors: &[Tensor], spm: &str, rotating: usize) -> String {
    let mut out = String::new();
    out.push_str("# Stream names for ");
    out.push_str(spm);
    out.push_str(&preamble(rotating));
    for (index, tensor) in tensors.iter().enumerate() {
        let (rows, cols) = tensor.stream_shape();
        let _ = writeln!(
            out,
            "{index}\t{}\t{rows}\t{cols}\t{}",
            tensor.name,
            u64::from(rows) * u64::from(cols)
        );
    }
    out
}

/// Total weights across `tensors`, counted as the streams will hold
/// them.
///
/// Uses [`Tensor::stream_shape`] rather than the raw shape so the
/// number agrees with what the `.spm` actually contains -- if the two
/// ever disagreed, this is the one a reader could check.
#[must_use]
pub fn total_weights(tensors: &[Tensor]) -> u64 {
    tensors
        .iter()
        .map(|t| {
            let (rows, cols) = t.stream_shape();
            u64::from(rows) * u64::from(cols)
        })
        .sum()
}

/// The commentary at the top of a sidecar.
///
/// Emitted with the table rather than left to documentation: a name
/// table found on its own should say what the rotating boundary means
/// and why the region is first.
fn preamble(rotating: usize) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "\n#\n\
         # .spm carries no names: the FPGA streams bytes in the order\n\
         # the directory declares and never needs them. This file is\n\
         # the host-side mapping and belongs beside that .spm.\n\
         #\n\
         # rotating-streams\t{rotating}\n\
         # Streams 0 to {rotating} exclusive are swept once per\n\
         # operation and rewound; everything after is read once into\n\
         # RAM. rewind() returns to stream 0, which is why the\n\
         # rotating region comes first.\n\
         # stream\tname\trows\tcols\telements\n"
    );
    out
}
