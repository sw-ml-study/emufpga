//! Text serialization of a golden case.
//!
//! Text, not a binary blob. These files exist to be read by a human
//! debugging a mismatch between software and hardware, and a diff that
//! names the row which changed is worth more than a compact encoding.
//!
//! Weights render as `-`, `0`, `+`, one character each, wrapped at the
//! row count -- so a column of the matrix reads down a column of the
//! file, matching the order the engine consumes them in.

mod fields;
mod parse;
mod render;

pub use fields::ParseError;
pub use parse::parse;
pub use render::render;
