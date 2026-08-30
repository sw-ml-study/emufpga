//! The operation descriptor and the datapath profile it names.

use core::fmt;

/// Each descriptor occupies a fixed 32 bytes.
pub const DESCRIPTOR_LEN: usize = 32;

/// The datapath profile: weight encoding, scale type and accumulator
/// width, named as one value.
///
/// These three co-vary in practice -- an FPGA is built for one
/// combination, not for a matrix of independent choices -- so the
/// format names the combination rather than three orthogonal fields.
/// Unknown discriminants are rejected rather than ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Two-bit ternary weights, `f32` group scales, `i32`
    /// accumulators. The only profile saga 1 implements.
    Ternary2F32I32,
}

impl Encoding {
    /// The on-disk discriminant.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Ternary2F32I32 => 1,
        }
    }

    /// Decodes an on-disk discriminant.
    ///
    /// # Errors
    /// Returns [`LayoutError::UnknownEncoding`] for any value this
    /// build does not implement.
    pub const fn from_code(code: u8) -> Result<Self, LayoutError> {
        match code {
            1 => Ok(Self::Ternary2F32I32),
            code => Err(LayoutError::UnknownEncoding { code }),
        }
    }
}

/// One streamed operation: a matrix and how to consume it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpDescriptor {
    /// Output count, M. The number of accumulators the engine needs.
    pub rows: u32,
    /// Input count, N. The number of activations held resident.
    pub cols: u32,
    /// Weights per scale group, in stream order.
    pub group_size: u32,
    /// Weight encoding, scale type and accumulator width.
    pub encoding: Encoding,
    /// Parallel weight lanes the stream is striped across.
    pub lane_count: u16,
}

/// A descriptor that could not be read, or does not describe a
/// consumable stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Fewer than [`DESCRIPTOR_LEN`] bytes were available.
    TooShort {
        /// Bytes available.
        available: usize,
    },
    /// The encoding discriminant is not one this build implements.
    UnknownEncoding {
        /// The offending discriminant.
        code: u8,
    },
    /// `group_size` was zero, which would divide by zero when
    /// counting groups.
    ZeroGroupSize,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { available } => {
                write!(f, "descriptor truncated: {available} bytes, need 32")
            }
            Self::UnknownEncoding { code } => {
                write!(f, "unknown .spm encoding profile {code}")
            }
            Self::ZeroGroupSize => write!(f, "group_size must be at least 1"),
        }
    }
}

impl core::error::Error for LayoutError {}
