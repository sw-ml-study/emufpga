//! Turning a dense f32 matrix into a ternary `.spm` file.
//!
//! # The quantization rule, stated rather than assumed
//!
//! Per-group **absmean**, the rule `BitNet` b1.58 uses. For each scale
//! group, in stream order:
//!
//! ```text
//! scale = mean(|w|) over the group
//! t     = clamp(round(w / scale), -1, +1)
//! ```
//!
//! Ties round half away from zero, which is what `f32::round` does.
//!
//! A group whose weights are all zero has `mean(|w|) == 0` and no
//! meaningful scale. It is written as `scale = 1.0` with every weight
//! `Zero`, which dequantizes back to zero exactly. Writing `0.0` would
//! also be correct arithmetically but would put a zero into the wire
//! format that the engine would multiply an activation by every
//! column, and a scale of zero is a value the hardware would rather
//! never see.
//!
//! This is a lossy transform and the loss is the point of the
//! architecture, so it is stated here and measured in the tests rather
//! than hidden behind a default. Nothing here silently picks a
//! rounding mode.
//!
//! # Consumption order
//!
//! Input is read row-major, the way a human writes a matrix. Output is
//! written column-major, the order the engine consumes
//! (docs/spm-format.md). [`quantize`] is where that transposition
//! happens, and it is the only place it happens.

mod emit;
mod matrix;
mod quantize;

pub use emit::write_spm;
pub use matrix::{Matrix, MatrixError, parse_matrix};
pub use quantize::{Quantized, quantize};
