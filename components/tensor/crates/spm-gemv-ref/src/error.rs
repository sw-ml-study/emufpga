//! Errors raised while running a streamed GEMV.

use spm_stream::StreamError;
use spm_stream_groups::GroupError;
use std::fmt;

/// A GEMV that could not be completed.
#[derive(Debug)]
pub enum GemvError {
    /// The parameter stream failed or was malformed.
    Group(GroupError),
    /// Fewer activations were supplied than the descriptor's `cols`.
    ///
    /// The activations are resident in RAM and known before the scan
    /// starts, so this is caught up front rather than part way
    /// through a stream that cannot be rewound.
    MissingActivations {
        /// Activations the descriptor requires per lane.
        needed: usize,
        /// Activations supplied.
        supplied: usize,
    },
    /// The stream declared no operations to run.
    NoStreams,
}

impl fmt::Display for GemvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Group(e) => write!(f, "{e}"),
            Self::MissingActivations { needed, supplied } => {
                write!(f, "need {needed} activations per lane, got {supplied}")
            }
            Self::NoStreams => write!(f, "parameter stream declares no operations"),
        }
    }
}

impl std::error::Error for GemvError {}

impl From<GroupError> for GemvError {
    fn from(value: GroupError) -> Self {
        Self::Group(value)
    }
}

impl From<StreamError> for GemvError {
    fn from(value: StreamError) -> Self {
        Self::Group(GroupError::Stream(value))
    }
}
