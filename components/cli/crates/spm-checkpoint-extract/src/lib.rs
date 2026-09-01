//! Checkpoint tensors to `.spm` consumption-order blobs.

mod transform;
mod write;

pub use write::{Encoding, Summary, extract};
