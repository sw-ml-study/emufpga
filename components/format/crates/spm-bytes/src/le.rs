//! Little-endian reads and writes over a byte slice.

/// Reads a `u16` at `offset`, or `None` if it would run past the end.
#[must_use]
pub fn read_u16(src: &[u8], offset: usize) -> Option<u16> {
    let raw = src.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([raw[0], raw[1]]))
}

/// Reads a `u32` at `offset`, or `None` if it would run past the end.
#[must_use]
pub fn read_u32(src: &[u8], offset: usize) -> Option<u32> {
    let raw = src.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

/// Writes a `u16` at `offset`.
///
/// # Panics
/// Panics if `dst` is too short. Callers size their buffers from the
/// format constants, so a short buffer is a bug in this crate rather
/// than bad input.
pub fn write_u16(dst: &mut [u8], offset: usize, value: u16) {
    dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Writes a `u32` at `offset`.
///
/// # Panics
/// Panics if `dst` is too short, for the same reason as [`write_u16`].
pub fn write_u32(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
