//! The f32 reference matmul and the error metrics that compare
//! against it.
//!
//! This is the "conventional implementation" side of every
//! correctness check in the project: the naive, obviously-correct
//! `y = Wx` that the streaming engine must reproduce. It makes no
//! attempt to be fast, and it must never be optimised into agreeing
//! with the thing it is checking.

mod matmul;
mod metrics;

pub use matmul::{DenseTernary, reference_gemv};
pub use metrics::{cosine_similarity, max_abs_error, mean_error};
