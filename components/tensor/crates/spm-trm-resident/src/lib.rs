//! The conventional path: TRM's forward pass with every weight
//! resident in RAM, addressed at random.
//!
//! This crate exists to be the thing [`spm_trm`] is compared against.
//! It computes the identical arithmetic in the identical order, so the
//! two agree bit for bit, and the only difference between them is
//! **where the weights live and how they are reached**.
//!
//! Deliberately a separate crate rather than a generic parameter on
//! [`spm_trm`]. Abstracting over "weight source" would mean a trait
//! that admits random access, and the streamed path's central
//! guarantee is that random access is not expressible in its types
//! (CLAUDE.md rule 1). A sibling crate keeps that guarantee intact and
//! makes the contrast explicit rather than hiding it behind a generic.
//!
//! What the comparison does NOT establish is in docs/results.md: this
//! resident path is the same scalar reference loop, not an optimised
//! GEMM, so it measures the cost of the streaming *mechanism* with the
//! arithmetic held fixed. It is not a race against a real inference
//! engine.

mod block;
mod recursion;
mod weights;

pub use block::ResidentLayer;
pub use recursion::forward;
pub use weights::ResidentWeights;
