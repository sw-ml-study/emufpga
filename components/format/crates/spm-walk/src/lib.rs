//! Where am I in the sequence of (stream, scale group)?
//!
//! The reader and the writer both walk a `.spm` file's groups in the
//! same order, and step 003's `WeightStream` walks it again. Keeping
//! the position logic in one place means the three cannot disagree
//! about where a stream ends -- a disagreement that would show up as
//! plausible-looking wrong numbers rather than an error.
//!
//! A [`Cursor`] moves forward only. It has no method to jump to a
//! given stream or group, for the same reason `SpmReader` has no seek.

#![no_std]

mod cursor;

pub use cursor::Cursor;
