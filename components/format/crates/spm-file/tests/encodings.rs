//! The encoding is consulted per stream, not per file.
//!
//! This is the property the whole step exists to establish. Before it,
//! every group sizing in the tree called `spm_codec::packed_len`,
//! which hardcodes ternary's two bits per weight -- so a file could
//! only ever hold one encoding, whatever its descriptors said.

use spm_codec::Ternary::{Minus, Plus, Zero};
use spm_codec_dense::{decode_into, dense_len, encode_into};
use spm_file::{SpmReader, SpmWriter};
use spm_layout::{Encoding, OpDescriptor};

fn descriptor(rows: u32, cols: u32, group_size: u32, encoding: Encoding) -> OpDescriptor {
    OpDescriptor {
        rows,
        cols,
        group_size,
        encoding,
        lane_count: 1,
    }
}

/// Writes an f32 group through the encoding-neutral path.
fn write_dense(writer: &mut SpmWriter, scale: f32, values: &[f32]) {
    let mut bytes = vec![0u8; dense_len(values.len())];
    encode_into(values, &mut bytes).expect("encode");
    writer
        .write_raw_group(scale, &bytes, values.len())
        .expect("write dense group");
}

#[test]
fn an_f32_stream_roundtrips_bit_for_bit() {
    // 2x2 = 4 weights, group of 4: one group, 16 bytes of payload.
    let descriptors = vec![descriptor(2, 2, 4, Encoding::F32)];
    let mut writer = SpmWriter::new(descriptors.clone());
    let values = [1.5f32, -0.25, core::f32::consts::PI, 0.0];
    write_dense(&mut writer, 1.0, &values);
    let bytes = writer.finish().expect("finish");

    let mut reader = SpmReader::parse(&bytes).expect("parse");
    let group = reader.next_group().expect("group").expect("ok");
    assert_eq!(group.count, 4);
    assert_eq!(group.packed.len(), 16, "f32 groups are 4 bytes per weight");
    let mut back = vec![0.0f32; 4];
    decode_into(group.packed, &mut back).expect("decode");
    for (a, b) in back.iter().zip(&values) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(reader.next_group().is_none());
}

#[test]
fn ternary_and_f32_streams_coexist_in_one_file() {
    // The proof that sizing is per stream. A ternary group of 4 is one
    // byte; an f32 group of 4 is sixteen. If the reader used a single
    // rule for the file, the second stream would be misaligned and the
    // values would come back as garbage rather than as an error.
    let descriptors = vec![
        descriptor(2, 2, 4, Encoding::Ternary2F32I32),
        descriptor(2, 2, 4, Encoding::F32),
    ];
    let mut writer = SpmWriter::new(descriptors.clone());
    writer
        .write_group(2.0, &[Plus, Zero, Minus, Plus])
        .expect("ternary group");
    let dense = [10.0f32, -20.0, 30.5, -40.25];
    write_dense(&mut writer, 1.0, &dense);
    let bytes = writer.finish().expect("finish");

    // 32 header + 64 directory + (4 + 1) ternary + (4 + 16) f32
    assert_eq!(bytes.len(), 32 + 64 + 5 + 20);

    let mut reader = SpmReader::parse(&bytes).expect("parse");
    let first = reader.next_group().expect("first").expect("ok");
    assert_eq!((first.stream, first.packed.len()), (0, 1));
    assert_eq!(first.packed[0], 0b01_11_00_01);

    let second = reader.next_group().expect("second").expect("ok");
    assert_eq!((second.stream, second.packed.len()), (1, 16));
    let mut back = vec![0.0f32; 4];
    decode_into(second.packed, &mut back).expect("decode");
    assert_eq!(back, dense);
}

#[test]
fn a_group_size_that_does_not_divide_works_for_both_encodings() {
    for (encoding, per_weight) in [(Encoding::Ternary2F32I32, 0), (Encoding::F32, 4)] {
        // 3x2 = 6 weights, group 4 -> groups of 4 and 2.
        let descriptors = vec![descriptor(3, 2, 4, encoding)];
        let mut writer = SpmWriter::new(descriptors.clone());
        if encoding == Encoding::F32 {
            write_dense(&mut writer, 1.0, &[1.0, 2.0, 3.0, 4.0]);
            write_dense(&mut writer, 1.0, &[5.0, 6.0]);
        } else {
            writer.write_group(1.0, &[Plus; 4]).expect("full group");
            writer.write_group(1.0, &[Minus; 2]).expect("short group");
        }
        let bytes = writer.finish().expect("finish");
        let mut reader = SpmReader::parse(&bytes).expect("parse");
        let lengths: Vec<(u32, usize)> = std::iter::from_fn(|| reader.next_group())
            .map(|g| {
                let g = g.expect("ok");
                (g.count, g.packed.len())
            })
            .collect();
        if per_weight == 4 {
            assert_eq!(lengths, vec![(4, 16), (2, 8)], "{encoding:?}");
        } else {
            // Ternary rounds up to a byte and pads: 4 weights -> 1 byte,
            // 2 weights -> 1 byte with the top four bits zero.
            assert_eq!(lengths, vec![(4, 1), (2, 1)], "{encoding:?}");
        }
    }
}

#[test]
fn a_payload_of_the_wrong_size_for_its_encoding_is_refused() {
    // The check that makes write_raw_group safe: bytes must match what
    // the stream's encoding implies for that many weights.
    let descriptors = vec![descriptor(2, 2, 4, Encoding::F32)];
    let mut writer = SpmWriter::new(descriptors);
    // 4 weights of f32 need 16 bytes; offer a ternary-sized payload.
    assert!(writer.write_raw_group(1.0, &[0u8; 1], 4).is_err());
}
