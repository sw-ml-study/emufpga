//! Resident activations, and the one place a multiply happens.
//!
//! docs/plan.md section 3 allows this explicitly: activations,
//! accumulators, scales and routing state may use ordinary
//! random-access memory. Only the parameter stream is restricted. For
//! a 70B model these are kilobytes against tens of gigabytes, and that
//! asymmetry is the entire architecture.
//!
//! Its own crate rather than a module inside the reference engine
//! because the saga 2 fabric engine needs exactly the same resident
//! state, and duplicating the scaling rule between the two would be a
//! good way to make them disagree.

mod activations;

pub use activations::Activations;
