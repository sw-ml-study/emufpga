//! Group-level encode and decode.

use crate::convert::{round, widen};

/// Bytes a group of `count` weights occupies. Two each.
#[must_use]
pub const fn bf16_len(count: usize) -> usize {
    count * 2
}

/// Writes `weights` as little-endian `bf16` into `dst`.
///
/// # Errors
/// Returns the bytes needed if `dst` is too short.
pub fn encode_into(weights: &[f32], dst: &mut [u8]) -> Result<usize, usize> {
    let needed = bf16_len(weights.len());
    if dst.len() < needed {
        return Err(needed);
    }
    for (value, slot) in weights.iter().zip(dst[..needed].chunks_exact_mut(2)) {
        slot.copy_from_slice(&round(*value).to_le_bytes());
    }
    Ok(needed)
}

/// Reads little-endian `bf16` from `src` into `dst`.
///
/// # Errors
/// Returns the bytes needed if `src` is too short.
pub fn decode_into(src: &[u8], dst: &mut [f32]) -> Result<(), usize> {
    let needed = bf16_len(dst.len());
    if src.len() < needed {
        return Err(needed);
    }
    for (slot, raw) in dst.iter_mut().zip(src[..needed].chunks_exact(2)) {
        *slot = widen(u16::from_le_bytes([raw[0], raw[1]]));
    }
    Ok(())
}
