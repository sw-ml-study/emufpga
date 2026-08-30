//! Errors raised while reading scale groups off a stream.

use spm_header::HeaderError;
use spm_layout::LayoutError;
use spm_stream::StreamError;
use std::fmt;

/// A stream whose `.spm` structure could not be read.
#[derive(Debug)]
pub enum GroupError {
    /// The underlying parameter stream failed or ended early.
    Stream(StreamError),
    /// The header was malformed.
    Header(HeaderError),
    /// A stream descriptor was malformed.
    Layout(LayoutError),
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "{e}"),
            Self::Header(e) => write!(f, "{e}"),
            Self::Layout(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GroupError {}

impl From<StreamError> for GroupError {
    fn from(value: StreamError) -> Self {
        Self::Stream(value)
    }
}

impl From<HeaderError> for GroupError {
    fn from(value: HeaderError) -> Self {
        Self::Header(value)
    }
}

impl From<LayoutError> for GroupError {
    fn from(value: LayoutError) -> Self {
        Self::Layout(value)
    }
}
