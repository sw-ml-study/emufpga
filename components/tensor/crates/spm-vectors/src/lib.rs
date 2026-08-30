//! Golden vectors: the regression suite later implementations are
//! validated against.
//!
//! A case is a ternary matrix, its group scales and an activation
//! vector, all derived from one seed. The saga 2 fabric simulator and
//! the saga 6 RTL both have to reproduce the same outputs from the
//! same `.spm` bytes, and a suite that cannot be regenerated from a
//! number is a suite that quietly rots.
//!
//! Reproducibility is a property, not an aspiration, so two choices
//! back it up. The generator is an explicit xorshift written out here
//! rather than a library RNG whose algorithm may change between
//! versions. And scales and activations are built from binary
//! fractions (`k / 16`), which are exactly representable in `f32`, so
//! the fixture contributes no rounding of its own and any error a
//! test measures belongs to the engine.
//!
//! Text serialization lives in `spm-vectors-text`.

mod draw;
mod generate;
mod model;

pub use generate::generate;
pub use model::GoldenCase;
