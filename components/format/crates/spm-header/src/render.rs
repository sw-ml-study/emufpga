//! Writing a header back to bytes.

use crate::model::{HEADER_LEN, Header, MAGIC};
use spm_bytes::{write_u16, write_u32};

/// Renders `header` to its fixed 32-byte on-disk form.
///
/// Reserved bytes are zero. The golden fixtures assert this, so a
/// future field cannot be added without the byte-layout test noticing.
#[must_use]
pub fn render(header: &Header) -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[..MAGIC.len()].copy_from_slice(&MAGIC);
    write_u16(&mut out, 8, header.version_major);
    write_u16(&mut out, 10, header.version_minor);
    out[12] = header.endianness.code();
    write_u32(&mut out, 16, header.stream_count);
    out
}
