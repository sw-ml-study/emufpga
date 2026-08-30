//! A file-backed parameter stream.
//!
//! Reads through a pair of buffers so that a later overlapped backend
//! -- a prefetch thread, or `io_uring` -- can be dropped in without any
//! change to [`spm_stream::WeightStream`] or to its consumers.
//!
//! **Today the refill is synchronous**: when the active buffer is
//! drained, the pair swaps and the inactive buffer is filled inline.
//! Behaviourally that is a single buffer. The two-slot structure is
//! here for the seam, not because IO is currently overlapped, and the
//! metrics in `spm-stream-metrics` will report `eta` honestly either
//! way. Overlapping the fill is deliberately left to the step that
//! needs it, because until the reference engine exists there is no
//! consumer whose consumption rate would make the overlap measurable.

mod buffer;
mod file;

pub use file::{DEFAULT_CAPACITY, FileWeightStream};
