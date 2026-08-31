//! Importing is pure framing: the bytes that go in come back out.
//!
//! HERMETIC ON PURPOSE. These build small checkpoints in the test
//! rather than depending on the real 27.3 MB download, so `cargo test`
//! is fast and works offline. The real checkpoint is a manual
//! verification recorded in docs/results.md, not a fixture -- weights
//! never enter this repository.

use spm_codec_dense::decode_into;
use spm_file::SpmReader;
use spm_import::{GROUP_SIZE, Tensor, assemble, parse_manifest, render_sidecar, total_weights};
use std::path::PathBuf;

/// The value this test expects at stream `index`, element `i`.
///
/// A ramp, so a misordered or misaligned stream is obvious rather
/// than plausible. Built from `u16` because every count here is small
/// and `u16 -> f32` is exact, which keeps the fixture free of any
/// rounding of its own.
fn ramp(index: usize, i: usize) -> f32 {
    let index = u16::try_from(index).expect("small");
    let i = u16::try_from(i).expect("small");
    f32::from(index) * 4096.0 + f32::from(i)
}

/// Writes blobs for `tensors` into a scratch dir, filling each with a
/// recognisable ramp so a misordered stream is obvious.
fn scratch(name: &str, tensors: &[Tensor]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("emufpga-import-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    for (index, tensor) in tensors.iter().enumerate() {
        let (rows, cols) = tensor.stream_shape();
        let values: Vec<f32> = (0..rows as usize * cols as usize)
            .map(|i| ramp(index, i))
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(&tensor.blob), bytes).expect("write blob");
    }
    dir
}

fn tensor(name: &str, shape: &[u32], blob: &str) -> Tensor {
    Tensor {
        name: name.into(),
        shape: shape.to_vec(),
        blob: blob.into(),
    }
}

/// Reads every stream back as f32, in directory order.
fn stream_back(bytes: &[u8]) -> Vec<Vec<f32>> {
    let mut reader = SpmReader::parse(bytes).expect("parse");
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); reader.descriptors.len()];
    while let Some(group) = reader.next_group() {
        let group = group.expect("group");
        assert!(
            (group.scale - 1.0).abs() < f32::EPSILON,
            "f32 scale must be the inert 1.0, got {}",
            group.scale
        );
        let mut values = vec![0.0f32; group.count as usize];
        decode_into(group.packed, &mut values).expect("decode");
        out[group.stream].extend(values);
    }
    out
}

#[test]
fn weights_survive_the_round_trip_bit_for_bit() {
    // Shapes covering: 2-D, a group size that divides exactly, one
    // that does not, and 1-D.
    let tensors = vec![
        tensor("a.weight", &[4, 8], "0.bin"),
        tensor("b.weight", &[2048, 1], "1.bin"),
        tensor("c.bias", &[3], "2.bin"),
        tensor("d.weight", &[7, 5], "3.bin"),
    ];
    let dir = scratch("roundtrip", &tensors);
    let bytes = assemble(&tensors, &dir).expect("assemble");

    let streams = stream_back(&bytes);
    assert_eq!(streams.len(), tensors.len());
    for (index, (values, tensor)) in streams.iter().zip(&tensors).enumerate() {
        let (rows, cols) = tensor.stream_shape();
        assert_eq!(
            values.len(),
            rows as usize * cols as usize,
            "{}",
            tensor.name
        );
        for (i, got) in values.iter().enumerate() {
            assert_eq!(
                got.to_bits(),
                ramp(index, i).to_bits(),
                "{} at {i}",
                tensor.name
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_tensor_larger_than_one_group_is_split_and_rejoined() {
    // GROUP_SIZE weights exactly, then one more, so the second tensor
    // has a short final group of one.
    let tensors = vec![
        tensor("exact", &[GROUP_SIZE, 1], "0.bin"),
        tensor("plus_one", &[GROUP_SIZE + 1, 1], "1.bin"),
    ];
    let dir = scratch("groups", &tensors);
    let bytes = assemble(&tensors, &dir).expect("assemble");
    let streams = stream_back(&bytes);
    assert_eq!(streams[0].len(), GROUP_SIZE as usize);
    assert_eq!(streams[1].len(), GROUP_SIZE as usize + 1);
    // The first weight of the short final group: tensor 1, element
    // GROUP_SIZE. If groups were rejoined in the wrong order or a
    // group boundary were miscounted, this is where it would show.
    let want = ramp(1, GROUP_SIZE as usize);
    assert_eq!(streams[1][GROUP_SIZE as usize].to_bits(), want.to_bits());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_blob_that_disagrees_with_its_shape_is_refused() {
    // Caught before anything is written. A short blob would otherwise
    // produce a .spm whose descriptors promise more weights than the
    // payload holds, and the reader would only notice at the end.
    let tensors = vec![tensor("a.weight", &[4, 8], "0.bin")];
    let dir = scratch("short", &tensors);
    std::fs::write(dir.join("0.bin"), [0u8; 16]).expect("truncate");
    let error = assemble(&tensors, &dir).expect_err("must be refused");
    let text = format!("{error}");
    assert!(text.contains("a.weight") && text.contains("128"), "{text}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_manifest_and_sidecar_agree_on_shapes() {
    let text = "# name\tshape\tdtype\tblob\telements\n\
                m.w\t6,512\tFloatStorage\t000.bin\t3072\n\
                m.b\t2\tFloatStorage\t001.bin\t2\n";
    let tensors = parse_manifest(text).expect("parse");
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].stream_shape(), (6, 512));
    // 1-D becomes a single column, so it streams in natural order.
    assert_eq!(tensors[1].stream_shape(), (2, 1));
    assert_eq!(total_weights(&tensors), 3074);

    let sidecar = render_sidecar(&tensors, "model.spm", 1);
    assert!(sidecar.contains("model.spm"));
    assert!(sidecar.contains("0\tm.w\t6\t512\t3072"), "{sidecar}");
    assert!(sidecar.contains("1\tm.b\t2\t1\t2"), "{sidecar}");
}

#[test]
fn malformed_manifest_lines_are_refused() {
    assert!(parse_manifest("only\ttwo\n").is_err());
    assert!(parse_manifest("n\tnot-a-number\tf\tb\n").is_err());
    assert!(parse_manifest("n\t0,4\tf\tb\n").is_err());
    // Comments and blanks are skipped rather than rejected.
    assert_eq!(parse_manifest("# header\n\n").expect("parse").len(), 0);
}
