//! Packing and unpacking ternary weights: four to a byte, LSB pair
//! first.
//!
//! Encode and decode live in one module because they are exact
//! inverses over the same bit layout. Splitting them would let a
//! change to the bit order update one side without the other.

use crate::error::CodecError;
use crate::model::{CODE_MASK, Ternary};

/// Weights packed into one byte.
const PER_BYTE: usize = 4;
/// Bits occupied by one weight.
const BITS: usize = 2;

/// Bytes needed to pack `count` weights.
///
/// Scale groups are byte-aligned by construction, so a count that is
/// not a multiple of four rounds up and the trailing bits of the last
/// byte are zero. Alignment costs a few bits per group and buys a
/// hardware decoder that never straddles a byte boundary.
#[must_use]
pub const fn packed_len(count: usize) -> usize {
    count.div_ceil(PER_BYTE)
}

/// Reads the raw two-bit code at stream position `index`.
///
/// Returns `None` past the end of `src`. This is a bounds check on a
/// byte buffer, not random access into a weight stream: callers walk
/// `index` forward from zero.
#[must_use]
pub fn code_at(src: &[u8], index: usize) -> Option<u8> {
    let byte = src.get(index / PER_BYTE)?;
    Some((byte >> ((index % PER_BYTE) * BITS)) & CODE_MASK)
}

/// Packs `weights` into `dst`, returning the number of bytes written.
///
/// The written range is zeroed first, so the padding bits of a short
/// final byte are deterministic. The golden fixtures depend on it.
///
/// # Errors
/// Returns [`CodecError::BufferTooSmall`] if `dst` is shorter than
/// [`packed_len`] of the input.
pub fn encode_into(weights: &[Ternary], dst: &mut [u8]) -> Result<usize, CodecError> {
    let needed = packed_len(weights.len());
    let available = dst.len();
    let out = dst
        .get_mut(..needed)
        .ok_or(CodecError::BufferTooSmall { needed, available })?;
    out.fill(0);
    for (index, weight) in weights.iter().enumerate() {
        out[index / PER_BYTE] |= weight.code() << ((index % PER_BYTE) * BITS);
    }
    Ok(needed)
}

/// Unpacks `dst.len()` weights from `src` into `dst`.
///
/// # Errors
/// Returns [`CodecError::BufferTooSmall`] if `src` holds fewer than
/// `dst.len()` weights, or [`CodecError::ReservedCode`] if the stream
/// contains the invalid `0b10` pattern.
pub fn decode_into(src: &[u8], dst: &mut [Ternary]) -> Result<(), CodecError> {
    let needed = packed_len(dst.len());
    let available = src.len();
    if available < needed {
        return Err(CodecError::BufferTooSmall { needed, available });
    }
    for (index, slot) in dst.iter_mut().enumerate() {
        let code = code_at(src, index).ok_or(CodecError::BufferTooSmall { needed, available })?;
        *slot = Ternary::from_code(code)?;
    }
    Ok(())
}
