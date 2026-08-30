//! Descriptor round-trip and rejection of unusable descriptors.

use spm_layout::{DESCRIPTOR_LEN, Encoding, LayoutError, OpDescriptor, parse, render};

fn descriptor() -> OpDescriptor {
    OpDescriptor {
        rows: 512,
        cols: 256,
        group_size: 64,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 8,
    }
}

#[test]
fn roundtrips_through_bytes() {
    let bytes = render(&descriptor());
    assert_eq!(bytes.len(), DESCRIPTOR_LEN);
    assert_eq!(parse(&bytes), Ok(descriptor()));
}

#[test]
fn reserved_bytes_are_zero() {
    let bytes = render(&descriptor());
    assert_eq!(bytes[13], 0);
    assert_eq!(&bytes[16..32], &[0u8; 16]);
}

#[test]
fn unknown_encoding_and_zero_group_size_are_refused() {
    let mut bytes = render(&descriptor());
    bytes[12] = 200;
    assert_eq!(
        parse(&bytes),
        Err(LayoutError::UnknownEncoding { code: 200 })
    );

    let mut bytes = render(&descriptor());
    bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(parse(&bytes), Err(LayoutError::ZeroGroupSize));
}

#[test]
fn short_input_is_refused() {
    let bytes = render(&descriptor());
    assert_eq!(
        parse(&bytes[..16]),
        Err(LayoutError::TooShort { available: 16 })
    );
}
