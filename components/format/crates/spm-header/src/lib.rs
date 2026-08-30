//! The fixed 32-byte `.spm` file header.
//!
//! Layout, all integers little-endian:
//!
//! | offset | size | field |
//! |--------|------|-------|
//! | 0      | 8    | magic `89 53 50 4D 0D 0A 1A 0A` |
//! | 8      | 2    | `version_major` |
//! | 10     | 2    | `version_minor` |
//! | 12     | 1    | `endianness` (0 = little) |
//! | 13     | 3    | reserved, zero |
//! | 16     | 4    | `stream_count` |
//! | 20     | 12   | reserved, zero |
//!
//! The magic follows the PNG convention. The high bit in the first
//! byte catches a transport that strips to 7 bits; `\r\n` catches CRLF
//! translation; `\x1a` stops a DOS `type`; the trailing `\n` catches
//! the reverse LF-to-CRLF conversion. A `.spm` file crosses three
//! implementations -- this repository, an RP2350 streamer and an FPGA
//! loader -- and silent mangling in transit would be very expensive to
//! diagnose from the far end.
//!
//! There is deliberately **no offset table**. A sequential reader has
//! no use for one, and an offset field is an invitation to seek. The
//! directory follows the header immediately and the payload follows
//! the directory immediately.

#![no_std]

mod model;
mod parse;
mod render;

pub use model::{Endianness, HEADER_LEN, Header, HeaderError, MAGIC, VERSION_MAJOR, VERSION_MINOR};
pub use parse::parse;
pub use render::render;
