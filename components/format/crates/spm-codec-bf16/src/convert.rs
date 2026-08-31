//! One value, each way.

/// Rounds one `f32` to `bf16`, round-to-nearest-**even**.
///
/// Truncating instead -- just dropping the low 16 bits -- is the
/// obvious implementation and is biased: it always rounds toward
/// zero, so a long accumulation drifts. The `+ 0x7FFF + lsb` form
/// adds half an ulp, plus one more when the retained bit is odd,
/// which breaks ties to even.
///
/// `NaN` is quieted rather than allowed to round into an infinity.
pub(crate) fn round(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return u16::try_from((bits >> 16) | 0x0040).unwrap_or(0x7FC0);
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    u16::try_from(rounded >> 16).unwrap_or(0)
}

/// Widens one `bf16` to `f32`. Exact: every `bf16` is an `f32`.
pub(crate) const fn widen(raw: u16) -> f32 {
    f32::from_bits((raw as u32) << 16)
}
