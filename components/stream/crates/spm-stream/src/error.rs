//! Errors raised while pulling bytes off a parameter stream.

use std::fmt;
use std::io;

/// A parameter stream that could not be advanced.
#[derive(Debug)]
pub enum StreamError {
    /// The backing store failed.
    Io(io::Error),
    /// The stream ended while a consumer still needed bytes.
    ///
    /// Distinct from a clean end of stream, which [`super::WeightStream::next_block`]
    /// reports as a short or zero-length read. This variant means a
    /// caller asked for a specific count and the store could not
    /// supply it, which for a `.spm` payload means the file is
    /// truncated.
    Truncated {
        /// Bytes the caller required.
        needed: usize,
        /// Bytes the stream could still supply.
        available: usize,
    },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "parameter stream io: {e}"),
            Self::Truncated { needed, available } => write!(
                f,
                "parameter stream truncated: needed {needed} bytes, {available} remain"
            ),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Truncated { .. } => None,
        }
    }
}

impl From<io::Error> for StreamError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
