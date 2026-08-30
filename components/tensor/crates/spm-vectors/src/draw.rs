//! The deterministic generator behind a golden case.

use spm_codec::Ternary;

/// Advances an xorshift64 state.
///
/// Written out rather than pulled from a crate: a golden suite that
/// changes when a dependency changes its algorithm is not golden.
pub(crate) const fn next_state(state: u64) -> u64 {
    let mut s = state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
}

/// Draws `count` ternary weights, advancing `state`.
pub(crate) fn weights(state: &mut u64, count: usize) -> Vec<Ternary> {
    (0..count)
        .map(|_| {
            *state = next_state(*state);
            match *state % 3 {
                0 => Ternary::Zero,
                1 => Ternary::Plus,
                _ => Ternary::Minus,
            }
        })
        .collect()
}

/// Draws `count` values of the form `k / 16 + offset`.
///
/// Sixteenths are exactly representable in `f32`, so the fixture adds
/// no rounding of its own to the reference computation.
pub(crate) fn fractions(state: &mut u64, count: usize, span: u64, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *state = next_state(*state);
            f32::from(u16::try_from(*state % span.max(1)).unwrap_or(0)) / 16.0 + offset
        })
        .collect()
}
