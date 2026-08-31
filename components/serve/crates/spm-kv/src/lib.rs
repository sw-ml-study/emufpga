//! Per-client attention state, and attending against it.
//!
//! This is the half of a serving engine that does **not** amortize.
//! One sweep of the weights serves every client at once, because a
//! weight is fetched once and applied to all of them. A KV cache
//! cannot be shared that way: each client has its own prefix, its own
//! position, and its own keys and values.
//!
//! So the two halves of a decode step scale in opposite directions.
//! Weight traffic is constant in the client count; attention work and
//! attention memory are linear in it. docs/results.md reports where
//! that crossover lands, and it is the same shape as the BDH lesson
//! -- what streaming saves on parameters, serving spends on state.

mod attend;
mod cache;

pub use attend::{attend_cached, rotate_at};
pub use cache::KvCache;
