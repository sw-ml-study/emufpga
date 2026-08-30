//! Little-endian integer primitives for the `.spm` wire format.
//!
//! Every multi-byte field in a `.spm` file is little-endian regardless
//! of host byte order, so a big-endian host and the FPGA read the same
//! bytes. Kept in its own crate because the header, the stream
//! directory and the operation descriptors all need it and none of
//! them should depend on each other to get it.

#![no_std]

mod le;

pub use le::{read_u16, read_u32, write_u16, write_u32};
