//! Malformed and incomplete files must fail loudly.
//!
//! A short or mis-declared `.spm` file is more dangerous than a
//! missing one: the engine would read plausible garbage off the end of
//! the payload and produce numbers rather than an error.

use spm_codec::Ternary;
use spm_file::{FileError, SpmReader, SpmWriter};
use spm_header::HeaderError;
use spm_layout::{Encoding, OpDescriptor};

fn descriptor() -> OpDescriptor {
    OpDescriptor {
        rows: 3,
        cols: 2,
        group_size: 4,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 1,
    }
}

#[test]
fn finishing_with_unwritten_streams_fails() {
    let mut writer = SpmWriter::new(vec![descriptor()]);
    writer
        .write_group(1.0, &[Ternary::Plus; 4])
        .expect("first group");
    // The short second group is still owed.
    assert_eq!(
        writer.finish(),
        Err(FileError::Incomplete {
            written: 0,
            declared: 1
        })
    );
}

#[test]
fn a_group_of_the_wrong_length_is_refused() {
    let mut writer = SpmWriter::new(vec![descriptor()]);
    assert_eq!(
        writer.write_group(1.0, &[Ternary::Plus; 3]),
        Err(FileError::GroupLen {
            expected: 4,
            offered: 3
        })
    );
}

#[test]
fn writing_past_the_last_declared_stream_is_refused() {
    let mut writer = SpmWriter::new(vec![descriptor()]);
    writer.write_group(1.0, &[Ternary::Plus; 4]).expect("first");
    writer.write_group(1.0, &[Ternary::Plus; 2]).expect("short");
    assert_eq!(
        writer.write_group(1.0, &[Ternary::Plus; 4]),
        Err(FileError::Incomplete {
            written: 1,
            declared: 1
        })
    );
}

#[test]
fn a_truncated_payload_is_detected_rather_than_read_past() {
    let mut writer = SpmWriter::new(vec![descriptor()]);
    writer.write_group(1.0, &[Ternary::Plus; 4]).expect("first");
    writer.write_group(0.5, &[Ternary::Plus; 2]).expect("short");
    let bytes = writer.finish().expect("finish");

    // Drop the final packed byte.
    let mut reader = SpmReader::parse(&bytes[..bytes.len() - 1]).expect("parse");
    assert!(reader.next_group().expect("first group").is_ok());
    assert_eq!(
        reader.next_group().expect("second group"),
        Err(FileError::PayloadTruncated {
            needed: 5,
            available: 4
        })
    );
}

#[test]
fn header_failures_propagate_unchanged() {
    let mut bytes = SpmWriter::new(vec![descriptor()])
        .finish()
        .unwrap_or_default();
    if bytes.is_empty() {
        bytes = vec![0u8; 32];
    }
    bytes[0] = 0;
    assert_eq!(
        SpmReader::parse(&bytes).map(|_| ()),
        Err(FileError::Header(HeaderError::BadMagic))
    );
}
