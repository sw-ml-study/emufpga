//! Errors raised while reading or writing a `.spm` file.

use core::fmt;
use spm_codec::CodecError;
use spm_header::HeaderError;
use spm_layout::LayoutError;

/// A `.spm` file that could not be read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileError {
    /// The header could not be read.
    Header(HeaderError),
    /// A descriptor could not be read.
    Layout(LayoutError),
    /// Packed weights could not be encoded or decoded.
    Codec(CodecError),
    /// The payload ended mid-group.
    PayloadTruncated {
        /// Bytes the group needed.
        needed: usize,
        /// Bytes remaining in the payload.
        available: usize,
    },
    /// A group was offered with the wrong number of weights. The
    /// layout fixes every group length in advance, so this means the
    /// writer and the descriptor disagree.
    GroupLen {
        /// Weights the layout expects for this group.
        expected: u32,
        /// Weights offered.
        offered: usize,
    },
    /// Fewer groups were written than the descriptors call for.
    Incomplete {
        /// Streams fully written.
        written: usize,
        /// Streams the directory declares.
        declared: usize,
    },
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(e) => write!(f, "{e}"),
            Self::Layout(e) => write!(f, "{e}"),
            Self::Codec(e) => write!(f, "{e}"),
            Self::PayloadTruncated { needed, available } => write!(
                f,
                "payload truncated: group needs {needed} bytes, {available} remain"
            ),
            Self::GroupLen { expected, offered } => {
                write!(f, "group expects {expected} weights, got {offered}")
            }
            Self::Incomplete { written, declared } => {
                write!(f, "only {written} of {declared} streams were written")
            }
        }
    }
}

impl core::error::Error for FileError {}

impl From<HeaderError> for FileError {
    fn from(value: HeaderError) -> Self {
        Self::Header(value)
    }
}

impl From<LayoutError> for FileError {
    fn from(value: LayoutError) -> Self {
        Self::Layout(value)
    }
}

impl From<CodecError> for FileError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}
