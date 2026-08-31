//! Four little-endian bytes per weight, in stream order.

/// Bytes one `f32` occupies on the wire.
const WIDTH: usize = 4;

/// Bytes needed for `count` dense `f32` weights.
#[must_use]
pub const fn dense_len(count: usize) -> usize {
    count * WIDTH
}

/// Writes `weights` into `dst`, returning bytes written.
///
/// # Errors
/// Returns `Err` with the required length if `dst` is too short.
pub fn encode_into(weights: &[f32], dst: &mut [u8]) -> Result<usize, usize> {
    let needed = dense_len(weights.len());
    if dst.len() < needed {
        return Err(needed);
    }
    for (slot, weight) in dst[..needed].chunks_exact_mut(WIDTH).zip(weights) {
        slot.copy_from_slice(&weight.to_le_bytes());
    }
    Ok(needed)
}

/// Reads `dst.len()` weights from `src` into `dst`.
///
/// Lossless: these are the same bytes the writer emitted, so a
/// round trip is bit-equal rather than merely close.
///
/// # Errors
/// Returns `Err` with the required length if `src` is too short.
pub fn decode_into(src: &[u8], dst: &mut [f32]) -> Result<(), usize> {
    let needed = dense_len(dst.len());
    if src.len() < needed {
        return Err(needed);
    }
    for (slot, raw) in dst.iter_mut().zip(src[..needed].chunks_exact(WIDTH)) {
        *slot = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    }
    Ok(())
}
