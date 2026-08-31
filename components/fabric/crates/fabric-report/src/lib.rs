//! Rendering a fabric run.
//!
//! Separate from the model so presentation cannot leak into
//! measurement, matching how `spm-bench` and `spm-bench-report` are
//! split.

mod render;

pub use render::render;
