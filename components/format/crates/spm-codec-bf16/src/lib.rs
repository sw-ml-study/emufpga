//! The `bfloat16` wire format: two bytes per weight, no packing.
//!
//! `bf16` is the top 16 bits of an `f32` -- same sign, same 8-bit
//! exponent, 8 mantissa bits instead of 24. That makes conversion a
//! shift in one direction and a rounding in the other, with no range
//! change and nothing that can overflow.
//!
//! It halves the parameter traffic against [`spm_codec_dense`], which
//! is the point: checkpoints ship in `bf16`, and widening them on
//! import doubles every byte this project is trying to save.

mod bf16;
mod convert;

pub use bf16::{bf16_len, decode_into, encode_into};
