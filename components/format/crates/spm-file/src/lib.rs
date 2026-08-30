//! Reading and writing whole `.spm` files.
//!
//! File structure, in the order a sequential reader meets it:
//!
//! ```text
//! [ header      32 bytes                    ]
//! [ descriptor  32 bytes ] x stream_count
//! [ payload                                 ]
//! ```
//!
//! and the payload, per stream in directory order, per scale group:
//!
//! ```text
//! [ scale  f32 little-endian, 4 bytes ]
//! [ packed weights, byte-aligned      ]
//! ```
//!
//! The scale arrives immediately before the weights it applies to, so
//! the engine never has to seek to fetch one.
//!
//! # No seek, all the way down
//!
//! [`SpmReader`] exposes exactly one way to reach weights:
//! [`SpmReader::next_group`], which moves forward and cannot be
//! rewound mid-stream. There is no index-by-position method, not even
//! a private one, because step 003 builds `WeightStream` on top of
//! this type and the guarantee has to hold at every layer. Header and
//! descriptors ARE read into RAM and randomly accessible -- they are
//! metadata, and docs/plan.md section 3 allows metadata in ordinary
//! memory. Only the parameter stream is restricted.
//!
//! [`next_group`] hands back the packed bytes rather than decoded
//! weights, matching the `TensorSink::consume(&[u8])` shape of the
//! abstract machine: the sink receives bytes off the wire and decodes
//! them itself, exactly as the FPGA does.
//!
//! [`next_group`]: SpmReader::next_group

mod error;
mod read;
mod write;

pub use error::FileError;
pub use read::{Group, SpmReader};
pub use write::SpmWriter;
