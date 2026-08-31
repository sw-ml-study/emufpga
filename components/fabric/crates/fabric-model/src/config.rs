//! The abstract knobs.

use core::fmt;

/// A fabric configuration. Every field is a pure abstraction -- none
/// of them is derived from, or checked against, any real part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FabricConfig {
    /// Weights the datapath consumes per cycle.
    pub weight_lanes: usize,
    /// Accumulator updates per weight per cycle.
    ///
    /// A weight applied to a batch of `b` takes `ceil(b / batch_width)`
    /// cycles in the accumulate stage. Setting this equal to the batch
    /// size models a fabric wide enough to update every lane at once.
    pub batch_width: usize,
    /// Depth of the weight FIFO, in bytes.
    pub fifo_bytes: usize,
    /// Bytes the parameter store delivers per cycle once streaming.
    pub fetch_bytes_per_cycle: usize,
    /// Cycles before the first byte arrives.
    ///
    /// Paid once at the start of a scan. A real store also pays it on
    /// every discontinuity, but the whole point of the architecture is
    /// that there are none.
    pub fetch_latency_cycles: u64,
}

/// Anything that stops a fabric run.
///
/// One error type for the crate rather than a config error and a
/// stream error kept apart. A caller does the same thing with all of
/// them -- report and stop -- and the split would have cost a module
/// this crate does not have room for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FabricError {
    /// A field that must be at least 1 was zero.
    MustBePositive {
        /// Which field.
        field: &'static str,
    },
    /// The FIFO cannot hold a single cycle's fetch.
    ///
    /// Not merely inefficient: the fetch stage could never complete a
    /// transfer, so the model would report cycles for a pipeline that
    /// cannot run.
    FifoSmallerThanFetch {
        /// FIFO depth in bytes.
        fifo_bytes: usize,
        /// Bytes per fetch cycle.
        fetch_bytes_per_cycle: usize,
    },
    /// The `.spm` stream could not be read.
    MalformedStream {
        /// What the stream layer said.
        detail: String,
    },
    /// The file declares no operations to run.
    NoStreams,
}

impl fmt::Display for FabricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MustBePositive { field } => write!(f, "{field} must be at least 1"),
            Self::FifoSmallerThanFetch {
                fifo_bytes,
                fetch_bytes_per_cycle,
            } => write!(
                f,
                "fifo_bytes ({fifo_bytes}) must hold at least one fetch ({fetch_bytes_per_cycle})"
            ),
            Self::MalformedStream { detail } => write!(f, "malformed .spm stream: {detail}"),
            Self::NoStreams => write!(f, "parameter stream declares no operations"),
        }
    }
}

impl std::error::Error for FabricError {}

impl FabricConfig {
    /// A configuration wide enough that nothing starves: one weight
    /// per cycle, the whole batch updated at once, and a store that
    /// keeps the FIFO full.
    ///
    /// Useful as a baseline, because a run against it isolates the
    /// datapath's own cost from any stall behaviour.
    #[must_use]
    pub const fn unconstrained(batch: usize) -> Self {
        Self {
            weight_lanes: 1,
            batch_width: batch,
            fifo_bytes: 4096,
            fetch_bytes_per_cycle: 64,
            fetch_latency_cycles: 0,
        }
    }

    /// Checks the configuration describes a pipeline that can run.
    ///
    /// # Errors
    /// Returns [`FabricError`] if a positive field is zero, or the
    /// FIFO cannot hold one cycle's fetch.
    pub fn validate(&self) -> Result<(), FabricError> {
        if self.weight_lanes == 0 {
            return Err(FabricError::MustBePositive {
                field: "weight_lanes",
            });
        }
        if self.batch_width == 0 {
            return Err(FabricError::MustBePositive {
                field: "batch_width",
            });
        }
        if self.fetch_bytes_per_cycle == 0 {
            return Err(FabricError::MustBePositive {
                field: "fetch_bytes_per_cycle",
            });
        }
        if self.fifo_bytes < self.fetch_bytes_per_cycle {
            return Err(FabricError::FifoSmallerThanFetch {
                fifo_bytes: self.fifo_bytes,
                fetch_bytes_per_cycle: self.fetch_bytes_per_cycle,
            });
        }
        Ok(())
    }
}

/// Wraps a stream-layer failure.
///
/// A free function rather than a `From` impl: the stream layer's error
/// types live in three different crates and are not all convertible,
/// and a one-line call site is what keeps `scan` inside its budget.
pub(crate) fn malformed(error: impl core::fmt::Display) -> FabricError {
    FabricError::MalformedStream {
        detail: error.to_string(),
    }
}
