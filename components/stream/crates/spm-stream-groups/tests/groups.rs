//! Groups read off a stream must match groups read from memory.
//!
//! `spm-file`'s `SpmReader` walks a whole file already in RAM.
//! `GroupStream` walks the same file as it arrives. They must agree, or
//! the streaming path is not testable against the in-memory one.

use spm_codec::{Ternary, decode_into};
use spm_file::{SpmReader, SpmWriter};
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

fn descriptor(rows: u32, cols: u32, group_size: u32) -> OpDescriptor {
    OpDescriptor {
        rows,
        cols,
        group_size,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 1,
    }
}

/// Builds a file whose weights cycle deterministically.
fn build(descriptors: &[OpDescriptor]) -> Vec<u8> {
    let mut writer = SpmWriter::new(descriptors.to_vec());
    let mut cursor = spm_walk::Cursor::new(descriptors);
    let mut tick = 0u32;
    while let Some(len) = cursor.group_len(descriptors) {
        let weights: Vec<Ternary> = (0..len)
            .map(|i| match (i + tick) % 3 {
                0 => Ternary::Zero,
                1 => Ternary::Plus,
                _ => Ternary::Minus,
            })
            .collect();
        writer
            .write_group(f32::from(u16::try_from(tick).unwrap_or(1)) + 0.5, &weights)
            .expect("write group");
        cursor.advance(descriptors);
        tick += 1;
    }
    writer.finish().expect("finish")
}

/// Every group as (stream, scale, decoded weights).
type Groups = Vec<(usize, f32, Vec<Ternary>)>;

fn via_reader(bytes: &[u8]) -> Groups {
    let mut reader = SpmReader::parse(bytes).expect("parse");
    let mut out = Vec::new();
    while let Some(group) = reader.next_group() {
        let group = group.expect("group");
        let mut weights = vec![Ternary::Zero; group.count as usize];
        decode_into(group.packed, &mut weights).expect("decode");
        out.push((group.stream, group.scale, weights));
    }
    out
}

fn via_stream(bytes: &[u8]) -> Groups {
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.to_vec())).expect("open");
    let mut out = Vec::new();
    while let Some(group) = groups.next_group() {
        let group = group.expect("group");
        let mut weights = vec![Ternary::Zero; group.count as usize];
        decode_into(group.packed, &mut weights).expect("decode");
        out.push((group.stream, group.scale, weights));
    }
    out
}

#[test]
fn streaming_and_in_memory_readers_agree() {
    let descriptors = [
        descriptor(3, 2, 4),
        descriptor(16, 4, 8),
        descriptor(1, 1, 64),
    ];
    let bytes = build(&descriptors);
    assert_eq!(via_stream(&bytes), via_reader(&bytes));
}

#[test]
fn metadata_is_read_off_the_front_of_the_stream() {
    // Header and directory arrive sequentially like everything else;
    // nothing seeks to an offset to find them.
    let descriptors = [descriptor(3, 2, 4), descriptor(2, 2, 2)];
    let bytes = build(&descriptors);
    let groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    assert_eq!(groups.header.stream_count, 2);
    assert_eq!(groups.descriptors, descriptors);
}

#[test]
fn zero_weight_streams_are_skipped_not_misread() {
    // A stream declaring no weights contributes no groups. If the
    // cursor rested on it, the reader would consume a scale that was
    // never written and read the next stream's bytes as its own.
    let descriptors = [descriptor(0, 4, 8), descriptor(2, 1, 2)];
    let bytes = build(&descriptors);
    let groups = via_stream(&bytes);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0, 1);
    assert_eq!(via_stream(&bytes), via_reader(&bytes));
}

#[test]
fn a_truncated_payload_is_reported_not_read_past() {
    let bytes = build(&[descriptor(3, 2, 4)]);
    let short = &bytes[..bytes.len() - 1];
    let mut groups = GroupStream::open(MemoryWeightStream::new(short.to_vec())).expect("open");
    assert!(groups.next_group().expect("first").is_ok());
    let error = groups.next_group().expect("second").expect_err("truncated");
    assert!(format!("{error}").contains("truncated"), "got {error}");
}

#[test]
fn declared_widths_account_for_every_payload_byte() {
    // POSTMORTEM 2, DEFECT 11. `smol-xcheck` computed a forward pass's
    // traffic as `weights * size_of::<f32>()`, which is right at f32
    // and reports double at bf16 -- erasing the whole result of the
    // step that added the profile.
    //
    // The general form of that bug is a width taken from anywhere but
    // the descriptor. This asserts the descriptors are the authority:
    // what they declare must be exactly what the file holds, for every
    // profile, so a traffic figure derived from them is derived from
    // the truth.
    for encoding in [Encoding::F32, Encoding::Bf16, Encoding::Ternary2F32I32] {
        let descriptors = vec![
            OpDescriptor {
                rows: 7,
                cols: 5,
                group_size: 8,
                encoding,
                lane_count: 1,
            },
            OpDescriptor {
                rows: 4,
                cols: 4,
                group_size: 8,
                encoding,
                lane_count: 1,
            },
        ];
        // A width test needs no values, only correctly sized groups.
        // `build` above writes ternary packing regardless of the
        // descriptor, so it cannot construct an f32 or bf16 file.
        let bytes = {
            let mut writer = SpmWriter::new(descriptors.clone());
            let mut cursor = spm_walk::Cursor::new(&descriptors);
            while let Some(len) = cursor.group_len(&descriptors) {
                let width = encoding.bytes_for(len as usize);
                writer
                    .write_raw_group(1.0, &vec![0u8; width], len as usize)
                    .expect("write raw group");
                cursor.advance(&descriptors);
            }
            writer.finish().expect("finish")
        };
        let declared: usize = descriptors
            .iter()
            .map(|d| encoding.bytes_for(d.rows as usize * d.cols as usize))
            .sum();

        let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
        let mut seen = 0usize;
        let mut weights = 0usize;
        while let Some(group) = groups.next_group() {
            let group = group.expect("group");
            assert_eq!(group.encoding, encoding, "view lost the encoding");
            assert_eq!(
                group.packed.len(),
                encoding.bytes_for(group.count as usize),
                "a group's bytes must match what its encoding declares"
            );
            seen += group.packed.len();
            weights += group.count as usize;
        }
        assert_eq!(weights, 7 * 5 + 4 * 4, "{encoding:?}: weight count");
        assert_eq!(
            seen, declared,
            "{encoding:?}: payload bytes ({seen}) must equal what the \
             descriptors declare ({declared})"
        );
    }
}
