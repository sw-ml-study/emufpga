//! Errors raised while packing or unpacking ternary weights.

use core::fmt;

/// A packing or unpacking failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The reserved `0b10` code was found in a packed stream.
    ReservedCode {
        /// The offending two-bit code.
        code: u8,
    },
    /// A value outside `-1..=1` was offered for packing.
    NotTernary {
        /// The offending value.
        value: i32,
    },
    /// The destination slice cannot hold the requested weights.
    BufferTooSmall {
        /// Bytes required.
        needed: usize,
        /// Bytes available.
        available: usize,
    },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedCode { code } => {
                write!(f, "reserved ternary code 0b{code:02b} in packed stream")
            }
            Self::NotTernary { value } => {
                write!(f, "value {value} is not ternary (expected -1, 0 or 1)")
            }
            Self::BufferTooSmall { needed, available } => {
                write!(f, "buffer too small: need {needed} bytes, have {available}")
            }
        }
    }
}

impl core::error::Error for CodecError {}
