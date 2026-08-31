//! Rendering a sweep as markdown.
//!
//! Separate from `spm-bench` so measurement and presentation cannot
//! grow into each other, and because step 008's `fit` needs its own
//! renderer against the same conventions.
//!
//! The caveats are emitted **with** the numbers rather than filed in a
//! document, because a table of ratios travels further than the
//! caveats that make it interpretable.

mod render;

pub use render::render;
