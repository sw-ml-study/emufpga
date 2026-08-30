//! Writing then reading a file must reproduce every group exactly.

use spm_codec::{Ternary, decode_into};
use spm_file::{SpmReader, SpmWriter};
use spm_layout::{Encoding, OpDescriptor};

fn descriptor(rows: u32, cols: u32, group_size: u32) -> OpDescriptor {
    OpDescriptor {
        rows,
        cols,
        group_size,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 1,
    }
}

/// Deterministic xorshift64, seeded per call; see spm-codec tests.
fn sequence(seed: u64, len: usize) -> Vec<Ternary> {
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

/// Writes every group of `descriptors`, returning the file and the
/// (scale, weights) pairs that went into it.
fn build(descriptors: &[OpDescriptor]) -> (Vec<u8>, Vec<(f32, Vec<Ternary>)>) {
    let mut writer = SpmWriter::new(descriptors.to_vec());
    let mut cursor = spm_walk::Cursor::new(descriptors);
    let mut written = Vec::new();
    let mut seed = 1u32;
    while let Some(len) = cursor.group_len(descriptors) {
        let weights = sequence(u64::from(seed), len as usize);
        let scale = f32::from(u16::try_from(seed).unwrap_or(u16::MAX)) * 0.25;
        writer.write_group(scale, &weights).expect("write group");
        written.push((scale, weights));
        cursor.advance(descriptors);
        seed += 1;
    }
    (writer.finish().expect("finish"), written)
}

#[test]
fn roundtrips_a_multi_stream_file() {
    let descriptors = [
        descriptor(3, 2, 4),
        descriptor(8, 4, 8),
        descriptor(1, 1, 64),
    ];
    let (bytes, written) = build(&descriptors);

    let mut reader = SpmReader::parse(&bytes).expect("parse");
    assert_eq!(reader.descriptors, descriptors);
    assert_eq!(reader.header.stream_count, 3);

    let mut read_back = Vec::new();
    while let Some(group) = reader.next_group() {
        let group = group.expect("group");
        let mut weights = vec![Ternary::Zero; group.count as usize];
        decode_into(group.packed, &mut weights).expect("decode");
        read_back.push((group.scale, weights));
    }
    assert_eq!(read_back, written);
}

#[test]
fn group_indices_report_the_stream_they_came_from() {
    // The stream index must be the one the group belongs to, not the
    // one the cursor moved on to afterwards.
    let descriptors = [descriptor(2, 1, 2), descriptor(2, 1, 2)];
    let (bytes, _) = build(&descriptors);
    let mut reader = SpmReader::parse(&bytes).expect("parse");
    let streams: Vec<usize> = std::iter::from_fn(|| reader.next_group())
        .map(|g| g.expect("group").stream)
        .collect();
    assert_eq!(streams, vec![0, 1]);
}

#[test]
fn a_short_final_group_survives_the_roundtrip() {
    // 6 weights at group_size 4: the second group holds 2 weights and
    // its packed byte carries 4 bits of padding.
    let descriptors = [descriptor(3, 2, 4)];
    let (bytes, written) = build(&descriptors);
    let mut reader = SpmReader::parse(&bytes).expect("parse");
    let groups: Vec<u32> = std::iter::from_fn(|| reader.next_group())
        .map(|g| g.expect("group").count)
        .collect();
    assert_eq!(groups, vec![4, 2]);
    assert_eq!(written[1].1.len(), 2);
}

#[test]
fn a_file_with_no_streams_is_valid_and_empty() {
    let bytes = SpmWriter::new(Vec::new()).finish().expect("finish");
    let mut reader = SpmReader::parse(&bytes).expect("parse");
    assert_eq!(reader.header.stream_count, 0);
    assert!(reader.next_group().is_none());
}
