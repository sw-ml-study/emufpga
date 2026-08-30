//! Ternary GEMV over a sequential parameter stream.
//!
//! The reference implementation of the whole architecture, and the
//! oracle the saga 2 fabric simulator and the saga 6 RTL are checked
//! against.
//!
//! # The inner loop has no multiplier
//!
//! That is the claim the hardware rests on, so the code is written to
//! make it checkable by eye. Per weight:
//!
//! ```text
//! code & NONZERO_BIT  == 0  ->  nothing happens
//! code & NEGATIVE_BIT != 0  ->  accumulator -= activation
//! otherwise                 ->  accumulator += activation
//! ```
//!
//! Those two masks are used directly rather than matching on
//! [`spm_codec::Ternary`], because in the fabric they are not a
//! decoded value at all -- they are two wires. Bit 0 is the
//! accumulator enable and bit 1 is the add/subtract select. Whoever
//! writes the RTL should be able to read this loop and see their
//! datapath.
//!
//! # Where the multiplies went
//!
//! Group scales are folded into the **activation**, not applied to
//! each weight. When the scan crosses into a new (group, column)
//! pair it recomputes `scale * x[j]` once for every batch lane; every
//! weight after that is a plain add or subtract. With
//! `group_size == rows` a group is exactly one column, so that is one
//! multiply per column against `rows` accumulate operations.
//!
//! # Batch is reuse, not throughput
//!
//! Each weight is applied to every lane before it is discarded, so
//! `Ps` equals the lane count. A zero weight still counts as an
//! application: it occupies a slot in the stream and a cycle in the
//! engine. Counting it keeps `Ps` a measure of **reuse** rather than
//! of sparsity, which is a separate axis the format does not yet
//! exploit.

mod datapath;
mod error;
mod gemv;

pub use error::GemvError;
pub use gemv::{GemvOutcome, run_gemv};
pub use spm_activations::Activations;
