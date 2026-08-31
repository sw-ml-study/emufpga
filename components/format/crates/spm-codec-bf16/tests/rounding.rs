//! Round-trip, and the rounding cases a truncating implementation
//! would get wrong.

use spm_codec_bf16::{bf16_len, decode_into, encode_into};

/// Bit patterns, not values. Exactness is the property under test, and
/// comparing bits states that directly -- it also distinguishes -0.0
/// from 0.0, which a float comparison does not.
fn bits(value: f32) -> u32 {
    value.to_bits()
}

fn round_trip(values: &[f32]) -> Vec<f32> {
    let mut bytes = vec![0u8; bf16_len(values.len())];
    encode_into(values, &mut bytes).expect("encode");
    let mut out = vec![0.0f32; values.len()];
    decode_into(&bytes, &mut out).expect("decode");
    out
}

#[test]
fn values_representable_in_bf16_survive_exactly() {
    // Anything whose low 16 mantissa bits are already zero is a bf16.
    let values = vec![0.0f32, 1.0, -1.0, 2.0, 0.5, -256.0, 1.5, 3.0];
    let got: Vec<u32> = round_trip(&values).iter().copied().map(bits).collect();
    let want: Vec<u32> = values.iter().copied().map(bits).collect();
    assert_eq!(got, want);
}

#[test]
fn rounding_is_to_nearest_even_not_toward_zero() {
    // THE POINT OF WRITING THE ROUNDING OUT. Truncation always rounds
    // toward zero, which biases a long accumulation. These two are
    // exact halfway cases: 1.0 + 2^-9 sits midway between 1.0 and
    // 1.0 + 2^-8, so it must round to the EVEN neighbour, 1.0.
    let half_up = f32::from_bits(0x3F80_8000); // 1 + 2^-8/2, tie
    let next = f32::from_bits(0x3F81_0000); // 1 + 2^-8, odd mantissa
    assert_eq!(
        bits(round_trip(&[half_up])[0]),
        bits(1.0),
        "tie must go to even"
    );

    // And a tie just above an odd value rounds UP, away from zero --
    // which a truncating implementation would never do.
    let tie_above_odd = f32::from_bits(0x3F81_8000);
    assert_eq!(
        bits(round_trip(&[tie_above_odd])[0]),
        0x3F82_0000,
        "tie above an odd mantissa must round up to even"
    );
    assert_eq!(
        bits(round_trip(&[next])[0]),
        bits(next),
        "exact value unchanged"
    );
}

#[test]
fn rounding_beats_truncation_on_a_long_sum() {
    // The consequence, stated as a measurement rather than a claim.
    // Truncation is biased toward zero, so summing many positive
    // values drifts low. Round-to-nearest does not.
    let values: Vec<f32> = (1u16..=2000)
        .map(|i| f32::from(i).mul_add(1e-4, 1.0))
        .collect();
    let exact: f32 = values.iter().sum();
    let coded: f32 = round_trip(&values).iter().sum();
    let truncated: f32 = values
        .iter()
        .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
        .sum();
    let rounded_error = (coded - exact).abs();
    let truncated_error = (truncated - exact).abs();
    assert!(
        rounded_error < truncated_error,
        "rounding ({rounded_error}) should beat truncation ({truncated_error})"
    );
}

#[test]
fn a_short_buffer_reports_what_it_needed() {
    let mut small = vec![0u8; 2];
    assert_eq!(encode_into(&[1.0, 2.0], &mut small), Err(4));
    let mut out = vec![0.0f32; 2];
    assert_eq!(decode_into(&[0, 0], &mut out), Err(4));
}

#[test]
fn bf16_is_exactly_half_the_size_of_f32() {
    // The reason this encoding exists.
    assert_eq!(bf16_len(1000), 2000);
    assert_eq!(bf16_len(1000) * 2, spm_codec_dense::dense_len(1000));
}
