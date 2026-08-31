//! `y = Wx` with `W` arriving off a stream.
//!
//! `spm-gemv-ref` runs the first stream of a file with ternary
//! weights. A real model needs stream *N* of *M*, in f32, and needs
//! the same answer a resident implementation would give.
//!
//! # Two implementations on purpose
//!
//! Every matmul here exists twice: [`resident`] reads weights from an
//! array, [`streamed`] pulls them off a [`spm_stream::WeightStream`].
//! They must agree **bit-exactly**, and the tests assert that rather
//! than a tolerance.
//!
//! That is not redundancy. The resident version is what any
//! conventional runtime does, and it is obviously correct. The
//! streamed version is the thing under test. Comparing them is how we
//! learn that moving compute to the weight stream changes nothing
//! about the answer -- which is the claim the whole project rests on,
//! and it would be worth very little if checked against a tolerance.
//!
//! Agreement is exact rather than approximate because the stream is
//! column-major: consecutive weights land on consecutive output rows,
//! so no accumulator ever sees its terms reordered.
//!
//! # What is streamed and what is not
//!
//! Weights stream. Activations, accumulators and every elementwise
//! operator stay resident, which docs/plan.md section 3 allows
//! explicitly -- the restriction is on the parameter store, not on
//! the working set. Trying to make a softmax sequential would be
//! missing the point.

mod resident;
mod stream;

pub use resident::{LinearError, resident};
pub use stream::streamed;
