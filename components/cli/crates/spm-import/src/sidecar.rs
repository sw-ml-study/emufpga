//! Host-facing views of the tensor list: the name table that
//! travels beside a `.spm`, and the summary a caller prints.

use crate::manifest::Tensor;
use std::fmt::Write;

/// Renders the stream-index-to-name mapping.
///
/// Text, tab separated, one line per stream, in directory order. Text
/// because the reader is a human debugging a mismatch, or a scheduler
/// in whatever language; a binary table would buy nothing at this size
/// and cost everyone a parser.
#[must_use]
pub fn render_sidecar(tensors: &[Tensor], spm: &str) -> String {
    let mut out = String::new();
    out.push_str("# Stream names for ");
    out.push_str(spm);
    out.push_str(
        "\n#\n\
         # .spm carries no names: the FPGA streams bytes in the order\n\
         # the directory declares and never needs them. This file is\n\
         # the host-side mapping and belongs beside that .spm.\n\
         # stream\tname\trows\tcols\telements\n",
    );
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
