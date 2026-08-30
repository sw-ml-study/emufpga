//! The reserved code and the buffer bounds must both be enforced.

use spm_codec::{CodecError, Ternary, decode_into, encode_into};

#[test]
fn reserved_code_is_rejected() {
    // 0b10 is "negative zero", which the encoder never emits. It is
    // left permanently invalid so the hardware decoder stays
    // combinational, and it doubles as a corruption check.
    let packed = [0b10u8];
    let mut decoded = [Ternary::Zero; 1];
    assert_eq!(
        decode_into(&packed, &mut decoded),
        Err(CodecError::ReservedCode { code: 0b10 })
    );
}

#[test]
fn encoding_into_a_short_buffer_fails() {
    let weights = [Ternary::Plus; 5];
    let mut packed = [0u8; 1];
    assert_eq!(
        encode_into(&weights, &mut packed),
        Err(CodecError::BufferTooSmall {
            needed: 2,
            available: 1
        })
    );
}

#[test]
fn decoding_from_a_short_buffer_fails() {
    let packed = [0u8; 1];
    let mut decoded = [Ternary::Zero; 5];
    assert_eq!(
        decode_into(&packed, &mut decoded),
        Err(CodecError::BufferTooSmall {
            needed: 2,
            available: 1
        })
    );
}

#[test]
fn non_ternary_values_are_refused_rather_than_rounded() {
    // Quantization happens before packing. Silently rounding here
    // would hide a bug in the quantizer.
    assert_eq!(
        Ternary::from_value(2),
        Err(CodecError::NotTernary { value: 2 })
    );
    assert_eq!(Ternary::from_value(-1), Ok(Ternary::Minus));
}
