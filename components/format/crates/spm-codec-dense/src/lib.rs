//! Dense `f32` weights on the wire.
//!
//! The counterpart to `spm-codec`, which packs ternary. There is no
//! packing here -- four little-endian bytes per weight, in stream
//! order -- so this crate exists for symmetry and for a name, not
//! because the arithmetic is hard. Having it means `spm-file` and the
//! engines can ask an encoding for its byte math rather than assuming
//! anyone's.
//!
//! # Why f32 at all
//!
//! The TRM checkpoint this saga targets is f32 and stays f32. That
//! makes the GPU comparison numerically clean by construction: both
//! sides run identical weights, so any divergence is a bug in the
//! engine rather than a quantization artifact. Ternary is not
//! abandoned -- it belongs to the  R1 1.58-bit quant, later,
//! where the model is genuinely ternary and residency genuinely
//! bites.
//!
//! # The scale field is inert here
//!
//! The `.spm` layout writes one scale per group. For ternary that
//! scale carries the group's magnitude; for f32 the weights already
//! carry their own, so the writer emits `1.0` and readers ignore it.
//! The field is not removed for this encoding on purpose: the group
//! structure is what makes the stream self-describing, and an
//! encoding that skipped it would need a reader of its own.

#![no_std]

mod dense;

pub use dense::{decode_into, dense_len, encode_into};
