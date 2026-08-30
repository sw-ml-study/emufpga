//! Header round-trip and the failures that must not be silent.

use spm_header::{
    Endianness, HEADER_LEN, Header, HeaderError, MAGIC, VERSION_MAJOR, parse, render,
};

#[test]
fn roundtrips_through_bytes() {
    let header = Header::new(7);
    let bytes = render(&header);
    assert_eq!(bytes.len(), HEADER_LEN);
    assert_eq!(parse(&bytes), Ok(header));
    assert_eq!(header.endianness, Endianness::Little);
}

#[test]
fn reserved_bytes_are_zero() {
    // Pinned so a future field cannot be added without the byte-layout
    // tests noticing.
    let bytes = render(&Header::new(1));
    assert_eq!(&bytes[13..16], &[0, 0, 0]);
    assert_eq!(&bytes[20..32], &[0u8; 12]);
    assert_eq!(&bytes[..8], &MAGIC);
}

#[test]
fn a_newer_major_version_is_refused() {
    // Misparsing a weight stream yields plausible numbers rather than
    // an obvious error, so this has to fail loudly.
    let mut bytes = render(&Header::new(1));
    bytes[8] = 99;
    assert_eq!(
        parse(&bytes),
        Err(HeaderError::UnsupportedVersion {
            found: 99,
            supported: VERSION_MAJOR
        })
    );
}

#[test]
fn bad_magic_and_short_input_are_distinguished() {
    let mut bytes = render(&Header::new(1));
    bytes[0] = 0;
    assert_eq!(parse(&bytes), Err(HeaderError::BadMagic));
    assert_eq!(
        parse(&bytes[..8]),
        Err(HeaderError::TooShort { available: 8 })
    );
}
