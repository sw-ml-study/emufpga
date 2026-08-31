//! Reading and writing the fixed 32-byte operation descriptor.

use crate::model::{DESCRIPTOR_LEN, Encoding, LayoutError, OpDescriptor};
use core::fmt;
use spm_bytes::{read_u16, read_u32, write_u16, write_u32};

/// Parses the descriptor at the start of `src`.
///
/// # Errors
/// Returns [`LayoutError`] if the slice is short, the encoding profile
/// is unknown, or `group_size` is zero.
pub fn parse(src: &[u8]) -> Result<OpDescriptor, LayoutError> {
    let available = src.len();
    if available < DESCRIPTOR_LEN {
        return Err(LayoutError::TooShort { available });
    }
    let group_size = read_u32(src, 8).ok_or(LayoutError::TooShort { available })?;
    if group_size == 0 {
        return Err(LayoutError::ZeroGroupSize);
    }
    Ok(OpDescriptor {
        rows: read_u32(src, 0).ok_or(LayoutError::TooShort { available })?,
        cols: read_u32(src, 4).ok_or(LayoutError::TooShort { available })?,
        group_size,
        encoding: Encoding::from_code(src[12])?,
        lane_count: read_u16(src, 14).ok_or(LayoutError::TooShort { available })?,
    })
}

/// Renders `descriptor` to its fixed 32-byte on-disk form.
///
/// Reserved bytes are zero, and the golden fixtures assert it.
#[must_use]
pub fn render(descriptor: &OpDescriptor) -> [u8; DESCRIPTOR_LEN] {
    let mut out = [0u8; DESCRIPTOR_LEN];
    write_u32(&mut out, 0, descriptor.rows);
    write_u32(&mut out, 4, descriptor.cols);
    write_u32(&mut out, 8, descriptor.group_size);
    out[12] = descriptor.encoding.code();
    write_u16(&mut out, 14, descriptor.lane_count);
    out
}

// LayoutError is a wire-decoding error, so its rendering lives beside
// the wire code rather than beside the type it reports on. That also
// keeps model.rs inside the four-function module budget, and
// docs/code_metrics.md prefers splitting by responsibility over
// splitting mechanically.
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { available } => {
                write!(f, "descriptor truncated: {available} bytes, need 32")
            }
            Self::UnknownEncoding { code } => {
                write!(f, "unknown .spm encoding profile {code}")
            }
            Self::ZeroGroupSize => write!(f, "group_size must be at least 1"),
        }
    }
}

impl core::error::Error for LayoutError {}
