//! Streaming the weights must change nothing about the answer.
//!
//! Bit-exact, not within tolerance. That is the claim the project
//! rests on, and a tolerance would let a real reordering hide inside
//! it. Exactness is available because the stream is column-major:
//! consecutive weights land on consecutive output rows, so no
//! accumulator sees its terms reordered.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_linear::{resident, streamed};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_walk::Cursor;

/// Deterministic xorshift, as elsewhere in this repository: a seeded
/// generator keeps the fixtures reproducible without a dependency.
fn weights(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Sixteenths, exactly representable, so the fixture adds no
            // rounding of its own to either implementation.
            f32::from(u16::try_from(state % 64).unwrap_or(0)) / 16.0 - 2.0
        })
        .collect()
}

/// Frames matrices into a `.spm`, in the order given.
fn to_spm(shapes: &[(u32, u32)], group_size: u32, all: &[Vec<f32>]) -> Vec<u8> {
    let descriptors: Vec<OpDescriptor> = shapes
        .iter()
        .map(|(rows, cols)| OpDescriptor {
            rows: *rows,
            cols: *cols,
            group_size,
            encoding: Encoding::F32,
            lane_count: 1,
        })
        .collect();
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    for matrix in all {
        let mut at = 0usize;
        while let Some(count) = cursor.group_len(&descriptors) {
            let count = count as usize;
            let mut bytes = vec![0u8; dense_len(count)];
            encode_into(&matrix[at..at + count], &mut bytes).expect("encode");
            writer
                .write_raw_group(1.0, &bytes, count)
                .expect("write group");
            cursor.advance(&descriptors);
            at += count;
            if at >= matrix.len() {
                break;
            }
        }
    }
    writer.finish().expect("finish")
}

#[test]
fn streamed_and_resident_agree_bit_for_bit() {
    // TRM's real shapes, so the agreement is tested where it will be
    // used rather than on a toy.
    let shapes = [(1536u32, 512u32), (512, 512), (3072, 512), (512, 1536)];
    for (index, (rows, cols)) in shapes.into_iter().enumerate() {
        let (rows, cols) = (rows as usize, cols as usize);
        let matrix = weights(index as u64 + 1, rows * cols);
        let activations = weights(1000 + index as u64, cols);

        let mut want = vec![0.0f32; rows];
        resident(&matrix, (rows, cols), (&activations, 1), &mut want).expect("resident");

        let shape = (
            u32::try_from(rows).expect("fits"),
            u32::try_from(cols).expect("fits"),
        );
        let bytes = to_spm(&[shape], 1024, &[matrix]);
        let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
        let mut got = vec![0.0f32; rows];
        let stream =
            streamed(&mut groups, (rows, cols), (&activations, 1), &mut got).expect("streamed");

        assert_eq!(stream, 0);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{rows}x{cols} row {i}");
        }
    }
}

#[test]
fn consecutive_streams_are_consumed_in_order() {
    // A sweep runs stream 0, then 1, then 2 -- there is no way to ask
    // for one by index, because there is no seek. This is what a
    // consumption-order layout buys: the order the file declares is
    // the order the model wants.
    let shapes = [(8u32, 4u32), (4, 8), (16, 2)];
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| weights(i as u64 + 7, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, 16, &matrices);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");

    for (index, (rows, cols)) in shapes.into_iter().enumerate() {
        let (rows, cols) = (rows as usize, cols as usize);
        let activations = weights(500 + index as u64, cols);
        let mut want = vec![0.0f32; rows];
        resident(&matrices[index], (rows, cols), (&activations, 1), &mut want).expect("resident");
        let mut got = vec![0.0f32; rows];
        let seen =
            streamed(&mut groups, (rows, cols), (&activations, 1), &mut got).expect("streamed");
        assert_eq!(seen, index, "streams must arrive in declared order");
        assert_eq!(got, want, "stream {index}");
    }
}

#[test]
fn a_group_size_that_does_not_divide_still_agrees() {
    // 7x5 = 35 weights at group 16: groups of 16, 16 and 3.
    let (rows, cols) = (7usize, 5usize);
    let matrix = weights(42, rows * cols);
    let activations = weights(43, cols);
    let mut want = vec![0.0f32; rows];
    resident(&matrix, (rows, cols), (&activations, 1), &mut want).expect("resident");

    let shape = (
        u32::try_from(rows).expect("fits"),
        u32::try_from(cols).expect("fits"),
    );
    let bytes = to_spm(&[shape], 16, &[matrix]);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let mut got = vec![0.0f32; rows];
    streamed(&mut groups, (rows, cols), (&activations, 1), &mut got).expect("streamed");
    for (a, b) in got.iter().zip(&want) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn a_stream_that_ends_early_is_refused() {
    let matrix = weights(9, 8 * 4);
    let bytes = to_spm(&[(8, 4)], 16, &[matrix]);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let activations = weights(10, 4);
    let mut got = vec![0.0f32; 16];
    // Claim a bigger matrix than the file holds.
    let error = streamed(&mut groups, (16, 4), (&activations, 1), &mut got).expect_err("must fail");
    assert!(format!("{error}").contains("stream ended after"), "{error}");
}

#[test]
fn batching_reuses_each_weight_without_changing_any_answer() {
    // The reuse the architecture is built on: one sweep of the weights
    // serves every position. Each position must get exactly the answer
    // it would have got alone -- batching is an efficiency, not an
    // approximation.
    let (rows, cols, positions) = (16usize, 8usize, 5usize);
    let matrix = weights(77, rows * cols);
    let mut batch = Vec::new();
    for p in 0..positions {
        batch.extend(weights(100 + p as u64, cols));
    }
    let shape = (
        u32::try_from(rows).expect("fits"),
        u32::try_from(cols).expect("fits"),
    );
    let bytes = to_spm(&[shape], 32, std::slice::from_ref(&matrix));
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let mut got = vec![0.0f32; positions * rows];
    streamed(&mut groups, (rows, cols), (&batch, positions), &mut got).expect("streamed");

    for p in 0..positions {
        let mut alone = vec![0.0f32; rows];
        let row = &batch[p * cols..(p + 1) * cols];
        resident(&matrix, (rows, cols), (row, 1), &mut alone).expect("resident");
        for (i, (a, b)) in got[p * rows..(p + 1) * rows].iter().zip(&alone).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "position {p} row {i}");
        }
    }
}
