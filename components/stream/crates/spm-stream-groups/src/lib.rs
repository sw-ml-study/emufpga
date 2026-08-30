//! Scale groups, read sequentially off any [`spm_stream::WeightStream`].
//!
//! `spm-file`'s `SpmReader` walks a `.spm` file already held in
//! memory. This crate does the same walk over a stream that is still
//! arriving, which is the case that matters: the whole premise is that
//! the parameter store is too large and too slow to hold in RAM.
//!
//! Header and stream directory are read off the front of the stream
//! into ordinary memory -- docs/plan.md section 3 allows metadata in
//! RAM. Everything after that is pulled a group at a time into a
//! single reusable buffer, so the resident parameter bytes are one
//! group, not one model. [`GroupStream::resident_parameter_bytes`]
//! reports that figure so `Rp` can be measured rather than asserted.
//!
//! [`GroupStream::next_group`] borrows the reader mutably for as long
//! as the returned group lives, so a consumer cannot hold two groups
//! at once. That is not an inconvenience to work around: it is the
//! same constraint the hardware has, where the buffer is reused as
//! soon as the engine has consumed it.

mod error;
mod groups;
mod open;

pub use error::GroupError;
pub use groups::{GroupStream, GroupView};
