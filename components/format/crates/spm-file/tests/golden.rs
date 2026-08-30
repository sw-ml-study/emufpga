//! The on-disk byte layout is a contract, not an implementation
//! detail.
//!
//! `tests/golden/tiny.spm` was written by hand from
//! docs/spm-format.md, not produced by the writer under test, so this
//! compares two independent readings of the specification. Three
//! implementations will eventually read this format -- this
//! repository, an RP2350 streamer and an FPGA loader -- and a silent
//! layout change would make them disagree without anyone noticing.
//!
//! If this test fails, either the writer regressed or the format
//! changed. A format change is a deliberate act: bump the version in
//! spm-header, regenerate the fixture, and say so in the commit.

use spm_codec::{Ternary, decode_into};
use spm_file::{SpmReader, SpmWriter};
use spm_layout::{Encoding, OpDescriptor};

/// The committed fixture: a 3x2 matrix, scale group of 4, so one full
/// group of four weights and a short final group of two.
const GOLDEN: &[u8] = include_bytes!("golden/tiny.spm");

fn fixture_descriptor() -> OpDescriptor {
    OpDescriptor {
        rows: 3,
        cols: 2,
        group_size: 4,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 1,
    }
}

#[test]
fn the_writer_reproduces_the_fixture_byte_for_byte() {
    use Ternary::{Minus, Plus, Zero};
    let mut writer = SpmWriter::new(vec![fixture_descriptor()]);
    writer
        .write_group(1.0, &[Plus, Zero, Minus, Plus])
        .expect("first group");
    writer
        .write_group(0.5, &[Minus, Zero])
        .expect("short group");
    let produced = writer.finish().expect("finish");

    assert_eq!(produced.len(), GOLDEN.len(), "file length changed");
    assert_eq!(produced, GOLDEN, "on-disk layout changed");
}

#[test]
fn the_fixture_parses_back_to_the_values_that_built_it() {
    use Ternary::{Minus, Plus, Zero};
    let mut reader = SpmReader::parse(GOLDEN).expect("parse");
    assert_eq!(reader.descriptors, vec![fixture_descriptor()]);

    let mut groups = Vec::new();
    while let Some(group) = reader.next_group() {
        let group = group.expect("group");
        let mut weights = vec![Zero; group.count as usize];
        decode_into(group.packed, &mut weights).expect("decode");
        groups.push((group.scale, weights));
    }
    assert_eq!(
        groups,
        vec![
            (1.0, vec![Plus, Zero, Minus, Plus]),
            (0.5, vec![Minus, Zero]),
        ]
    );
}

#[test]
fn the_fixture_matches_the_documented_offsets() {
    // Spot-checks that a human reading docs/spm-format.md can follow.
    assert_eq!(
        &GOLDEN[..8],
        &[0x89, b'S', b'P', b'M', 0x0D, 0x0A, 0x1A, 0x0A]
    );
    assert_eq!(&GOLDEN[16..20], &1u32.to_le_bytes()); // stream_count
    assert_eq!(&GOLDEN[32..36], &3u32.to_le_bytes()); // rows
    assert_eq!(&GOLDEN[36..40], &2u32.to_le_bytes()); // cols
    assert_eq!(&GOLDEN[40..44], &4u32.to_le_bytes()); // group_size
    assert_eq!(GOLDEN[44], 1); // encoding profile
    assert_eq!(&GOLDEN[64..68], &1.0f32.to_le_bytes()); // first scale
    assert_eq!(GOLDEN[68], 0b01_11_00_01); // +, 0, -, + packed LSB first
    assert_eq!(&GOLDEN[69..73], &0.5f32.to_le_bytes()); // second scale
    assert_eq!(GOLDEN[73], 0b00_11); // -, 0 with padding bits zero
}
