//! Turning an extracted checkpoint into a `.spm` file.
//!
//! Input is what `scripts/extract-checkpoint` produces: a directory of
//! raw little-endian `f32` blobs and a `manifest.tsv` naming them.
//! Those bytes are **already** the `.spm` f32 encoding's wire format,
//! so this crate is pure framing -- it never touches a weight value.
//!
//! # Why the split is here
//!
//! A `.pt` is a ZIP holding a Python pickle. Reading one in Rust means
//! a ZIP reader plus a pickle VM of roughly 25 opcodes, written once
//! and then retired -- the `DeepSeek` R1 quant this project builds
//! toward is GGUF, not pickle. The extractor is Python for that
//! reason, and everything after it is Rust.
//!
//! # Names live in a sidecar, not in the container
//!
//! `.spm` has no name field and is not getting one. Names are a host
//! concern: the importer and a scheduler need them, the FPGA never
//! does -- it streams bytes in the order the directory declares.
//! Putting names in the container would make every consumer carry
//! metadata that only one of them uses.
//!
//! The sidecar is written beside the `.spm` and the two belong
//! together. Losing it costs you the mapping from stream index to
//! tensor name; it costs you nothing about the weights.

mod assemble;
mod manifest;
mod sidecar;

pub use assemble::{GROUP_SIZE, assemble, descriptors};
pub use manifest::{ImportError, Tensor, parse_manifest};
pub use sidecar::{render_sidecar, total_weights};
