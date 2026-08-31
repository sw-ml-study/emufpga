//! One sweep of the weights, many clients.
//!
//! The claim this crate exists to demonstrate: a decode step reads the
//! model **once** and produces one token for every waiting client, so
//! weight bytes per generated token fall as `1/N`.
//!
//! That is not an optimisation bolted on; it is the schedule the
//! parameter stream forces. There is no seek, so a weight that has
//! gone past cannot be fetched again within a step -- which means
//! every client that wants that weight has to be served while it is
//! here. Serving concurrently is the only way to read the model once.
//!
//! **Where the two halves diverge.** The streamed matmuls batch across
//! clients for free: `positions = clients`, and a weight is applied to
//! all of them before it is discarded. Attention does not batch at
//! all -- each client has its own cache, its own position, its own
//! prefix -- but it carries no weights, so it is resident work.
//! docs/plan.md section 3's asymmetry, finally load-bearing.
//!
//! What this does **not** claim is a throughput win. The engine is
//! still scalar reference code. What amortizes is the traffic, which
//! is a property of the schedule rather than of the arithmetic.

mod inner;
mod session;
mod step;

pub use session::{Client, Scratch, StepReport};
pub use step::decode_step;
