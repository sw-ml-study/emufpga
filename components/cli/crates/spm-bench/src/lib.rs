//! The batch-amortization sweep.
//!
//! docs/plan.md calls this Experiment 1E and the make-or-break
//! measurement of saga 1: storage bandwidth is fixed, so reuse is the
//! only free variable, and the question is how far reuse goes before
//! the tensor engine becomes the limit instead of the store.
//!
//! # What the sweep can and cannot tell you
//!
//! Be careful with `eta` here. `spm-stream-file` does not overlap IO
//! yet -- its two buffer slots refill synchronously -- so
//! `storage_time` and `compute_time` **partition** wall clock rather
//! than overlapping. `eta = storage_time / compute_time` therefore
//! measures a serial pipeline: the ratio of time this program spent
//! fetching to time it spent computing. That is a real result about
//! this implementation. It is **not** a proxy for what an FPGA would
//! do, where fetch and compute run concurrently by construction.
//!
//! What the number does support is the direction and the crossing
//! point: compute time grows with batch size while storage time does
//! not, so there is a batch size beyond which more reuse buys nothing
//! because the engine is already saturated.
//!
//! # Timer overhead is measured, not assumed
//!
//! The engine timestamps every scale group, twice. At small group
//! sizes that overhead is comparable to the work being timed, which
//! would silently inflate `storage_time`. [`timer_overhead`] measures
//! the cost of a timestamp pair on the machine actually running the
//! sweep, so a report can say whether a difference it shows is
//! resolvable at all.

mod model;
mod run;
mod setup;

pub use model::{Backend, BenchRow, Crossover, Sweep, timer_overhead};
pub use run::run_sweep;
