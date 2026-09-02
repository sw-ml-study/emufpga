#[test]
fn f32_known_values_and_rejects_partial_value() {
    let bytes = [0, 0, 128, 63, 0, 0, 32, 192];
    assert_eq!(spm_gguf::decode_f32(&bytes).unwrap(), [1.0, -2.5]);
    assert!(spm_gguf::decode_f32(&bytes[..7]).is_err());
}

#[test]
fn q6_k_zero_code_with_unit_scale_is_minus_32() {
    let mut block = [0_u8; 210];
    block[192..208].fill(1);
    block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    assert_eq!(spm_gguf::decode_q6_k(&block).unwrap(), vec![-32.0; 256]);
}

#[test]
fn q6_k_lane_bit_packing_matches_reference_layout() {
    let mut block = [0_u8; 210];
    block[192..208].fill(1);
    block[208..210].copy_from_slice(&0x4000_u16.to_le_bytes());
    block[0] = 0xa5;
    block[32] = 0xc7;
    block[128] = 0b11_10_01_00;
    let decoded = spm_gguf::decode_q6_k(&block).unwrap();
    let actual = [decoded[0], decoded[32], decoded[64], decoded[96]].map(f32::to_bits);
    let expected = [-54.0_f32, -18.0, 20.0, 56.0].map(f32::to_bits);
    assert_eq!(actual, expected);
}

#[test]
fn q6_k_rejects_partial_block() {
    assert!(spm_gguf::decode_q6_k(&[0; 209]).is_err());
}
