//! The streamed matmul must decode by the descriptor, not by habit.
//!
//! Before `GroupView` carried its encoding, `take` called the f32
//! codec unconditionally: a bf16 stream would have been read as f32
//! and produced plausible garbage with no error anywhere. The
//! discriminant was written into every descriptor and read by nobody,
//! which is the failure mode docs/postmortem-1.md keeps finding.

use spm_codec_bf16::encode_into as encode_bf16;
use spm_codec_dense::{dense_len, encode_into as encode_f32};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_linear::{resident, streamed};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

/// Frames one matrix at the given encoding.
fn spm(weights: &[f32], shape: (u32, u32), encoding: Encoding) -> Vec<u8> {
    let descriptor = OpDescriptor {
        rows: shape.0,
        cols: shape.1,
        group_size: 8,
        encoding,
        lane_count: 1,
    };
    let mut writer = SpmWriter::new(vec![descriptor]);
    for chunk in weights.chunks(8) {
        let mut bytes = vec![0u8; encoding.bytes_for(chunk.len())];
        match encoding {
            Encoding::Bf16 => encode_bf16(chunk, &mut bytes).map(|_| ()),
            _ => encode_f32(chunk, &mut bytes).map(|_| ()),
        }
        .expect("encode");
        writer
            .write_raw_group(1.0, &bytes, chunk.len())
            .expect("write");
    }
    writer.finish().expect("finish")
}

fn sweep(bytes: Vec<u8>, shape: (usize, usize), activations: &[f32]) -> Vec<f32> {
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let mut out = vec![0.0f32; shape.0];
    streamed(&mut groups, shape, (activations, 1), &mut out).expect("streamed");
    out
}

#[test]
fn a_bf16_stream_agrees_with_a_resident_matmul() {
    // Values chosen to be exactly representable in bf16, so the only
    // thing under test is that the bytes were read correctly.
    let weights: Vec<f32> = (0u16..24).map(|i| f32::from(i % 8) - 4.0).collect();
    let activations = vec![1.0f32, -2.0, 0.5, 4.0, -1.0, 2.0, -0.25, 8.0];
    let shape = (3usize, 8usize);
    let mut want = vec![0.0f32; shape.0];
    resident(&weights, shape, (&activations, 1), &mut want).expect("resident");

    let got = sweep(spm(&weights, (3, 8), Encoding::Bf16), shape, &activations);
    assert_eq!(got, want, "bf16 stream disagreed with a resident matmul");
}

#[test]
fn f32_and_bf16_agree_when_the_values_fit_in_bf16() {
    let weights: Vec<f32> = (0u16..24).map(|i| f32::from(i % 8) - 4.0).collect();
    let activations = vec![1.0f32, -2.0, 0.5, 4.0, -1.0, 2.0, -0.25, 8.0];
    let shape = (3usize, 8usize);
    let dense = sweep(spm(&weights, (3, 8), Encoding::F32), shape, &activations);
    let narrow = sweep(spm(&weights, (3, 8), Encoding::Bf16), shape, &activations);
    assert_eq!(
        dense, narrow,
        "the two profiles should agree on bf16 values"
    );
}

#[test]
fn bf16_really_is_narrower_and_the_two_files_differ() {
    // Guards against the bf16 path being secretly f32. Two things must
    // hold: the file is smaller, and a value needing more than 8
    // mantissa bits comes back rounded rather than exact. If either
    // fails, the encoding is not doing anything.
    let weights: Vec<f32> = (0u16..24)
        .map(|i| f32::from(i).mul_add(0.001, 1.0))
        .collect();
    let dense = spm(&weights, (3, 8), Encoding::F32);
    let narrow = spm(&weights, (3, 8), Encoding::Bf16);
    assert!(
        narrow.len() < dense.len(),
        "bf16 file ({}) should be smaller than f32 ({})",
        narrow.len(),
        dense.len()
    );

    let activations = vec![1.0f32; 8];
    let shape = (3usize, 8usize);
    let wide = sweep(dense, shape, &activations);
    let thin = sweep(narrow, shape, &activations);
    assert_ne!(
        wide, thin,
        "values needing 24 mantissa bits must be rounded by bf16; \
         equality here would mean the bf16 path is still f32"
    );
    for (a, b) in wide.iter().zip(&thin) {
        assert!((a - b).abs() < 0.05, "bf16 error too large: {a} vs {b}");
    }
}

#[test]
fn the_group_view_reports_the_encoding_it_was_written_with() {
    for encoding in [Encoding::F32, Encoding::Bf16] {
        let bytes = spm(&[1.0, 2.0, 3.0], (3, 1), encoding);
        let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
        let group = groups.next_group().expect("group").expect("ok");
        assert_eq!(group.encoding, encoding, "descriptor and view disagree");
        assert_eq!(group.packed.len(), encoding.bytes_for(3));
    }
    let _ = dense_len(1);
}
