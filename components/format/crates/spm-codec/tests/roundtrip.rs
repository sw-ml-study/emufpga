//! Encoding and decoding must be exact inverses over every input.

use spm_codec::{Ternary, code_at, decode_into, encode_into, packed_len};

/// Deterministic xorshift64. A seeded generator keeps the property
/// tests reproducible without pulling in a dependency, which matters
/// for a crate that has to stay `no_std` for the RP2350 front.
fn ternary_sequence(seed: u64, len: usize) -> Vec<Ternary> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match state % 3 {
                0 => Ternary::Zero,
                1 => Ternary::Plus,
                _ => Ternary::Minus,
            }
        })
        .collect()
}

#[test]
fn roundtrips_every_length_up_to_two_hundred() {
    // Lengths that are not multiples of four exercise the padding
    // bits of a short final byte.
    for len in 0..200 {
        let weights = ternary_sequence(len as u64 + 1, len);
        let mut packed = vec![0u8; packed_len(len)];
        let written = encode_into(&weights, &mut packed).expect("encode");
        assert_eq!(written, packed_len(len), "len {len}");

        let mut decoded = vec![Ternary::Zero; len];
        decode_into(&packed, &mut decoded).expect("decode");
        assert_eq!(decoded, weights, "len {len}");
    }
}

#[test]
fn padding_bits_of_a_short_final_byte_are_zero() {
    // The golden fixtures pin whole bytes, so padding must be
    // deterministic rather than whatever was in the buffer.
    let weights = [Ternary::Minus];
    let mut packed = [0xFFu8; 1];
    encode_into(&weights, &mut packed).expect("encode");
    assert_eq!(packed[0], Ternary::Minus.code(), "high 6 bits must be zero");
}

#[test]
fn weights_pack_four_to_a_byte_lsb_pair_first() {
    let weights = [Ternary::Plus, Ternary::Zero, Ternary::Minus, Ternary::Plus];
    let mut packed = [0u8; 1];
    encode_into(&weights, &mut packed).expect("encode");
    // 0b01 | 0b00<<2 | 0b11<<4 | 0b01<<6
    assert_eq!(packed[0], 0b01_11_00_01);
    for (index, weight) in weights.iter().enumerate() {
        assert_eq!(code_at(&packed, index), Some(weight.code()));
    }
}

#[test]
fn codes_encode_enable_and_sign_as_separate_bits() {
    // Bit 0 is the accumulator enable, bit 1 the subtract select.
    // The fabric wires these directly, so the mapping is load-bearing.
    assert_eq!(Ternary::Zero.code() & spm_codec::NONZERO_BIT, 0);
    assert_eq!(
        Ternary::Plus.code() & spm_codec::NONZERO_BIT,
        spm_codec::NONZERO_BIT
    );
    assert_eq!(
        Ternary::Minus.code() & spm_codec::NONZERO_BIT,
        spm_codec::NONZERO_BIT
    );
    assert_eq!(Ternary::Plus.code() & spm_codec::NEGATIVE_BIT, 0);
    assert_eq!(
        Ternary::Minus.code() & spm_codec::NEGATIVE_BIT,
        spm_codec::NEGATIVE_BIT
    );
}
