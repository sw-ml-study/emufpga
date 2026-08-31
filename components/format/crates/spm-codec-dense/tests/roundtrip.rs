//! Dense f32 is lossless, so round trips are bit-equal.

use spm_codec_dense::{decode_into, dense_len, encode_into};

#[test]
fn roundtrips_bit_for_bit() {
    // Values chosen to include the awkward ones: zero, negative zero,
    // subnormal, infinity, and a value with a full mantissa. If any of
    // these survived only "approximately", the encoding would be
    // lying about being lossless.
    let weights = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        core::f32::consts::PI,
        -123.456_79,
    ];
    let mut bytes = vec![0u8; dense_len(weights.len())];
    assert_eq!(
        encode_into(&weights, &mut bytes),
        Ok(dense_len(weights.len()))
    );

    let mut back = vec![0.0f32; weights.len()];
    decode_into(&bytes, &mut back).expect("decode");
    for (a, b) in back.iter().zip(&weights) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a} != {b} bitwise");
    }
}

#[test]
fn nan_survives_as_the_same_nan() {
    // NaN != NaN, so this can only be checked bitwise -- and a codec
    // that normalised NaN payloads would silently alter weights.
    let weights = [f32::NAN];
    let mut bytes = vec![0u8; dense_len(1)];
    encode_into(&weights, &mut bytes).expect("encode");
    let mut back = [0.0f32];
    decode_into(&bytes, &mut back).expect("decode");
    assert_eq!(back[0].to_bits(), f32::NAN.to_bits());
}

#[test]
fn byte_layout_is_little_endian() {
    // Pinned because the format says little-endian regardless of host,
    // and this crate is the only place that decides it for f32.
    let mut bytes = vec![0u8; 4];
    encode_into(&[1.0f32], &mut bytes).expect("encode");
    assert_eq!(bytes, vec![0x00, 0x00, 0x80, 0x3F]);
}

#[test]
fn short_buffers_report_what_was_needed() {
    let mut small = vec![0u8; 4];
    assert_eq!(encode_into(&[1.0f32, 2.0], &mut small), Err(8));
    let mut back = vec![0.0f32; 2];
    assert_eq!(decode_into(&small, &mut back), Err(8));
    assert_eq!(dense_len(0), 0);
}
