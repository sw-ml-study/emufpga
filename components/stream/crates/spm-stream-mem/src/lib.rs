//! An in-memory parameter stream.
//!
//! Used by tests and by the step 004 reference engine. Backed by a
//! byte slice, but deliberately exposing only [`spm_stream::WeightStream`]
//! -- a consumer cannot tell it apart from a file, which is what makes
//! it usable as the reference implementation that file-backed streams
//! must match.

mod mem;

pub use mem::MemoryWeightStream;
