//! The cycle counts are a model. The arithmetic is not.
//!
//! Agreement with the CPU reference must be BIT-EXACT, not
//! within-tolerance. The stream is column-major, so `weight_lanes`
//! consecutive weights land on `weight_lanes` different accumulators
//! and no accumulator ever sees a reordered summation. If that ever
//! stops holding, it is a finding to chase rather than a tolerance to
//! widen.

use fabric_model::{FabricConfig, run_fabric};
use spm_activations::Activations;
use spm_codec::Ternary;
use spm_file::SpmWriter;
use spm_gemv_ref::run_gemv;
use spm_stream_mem::MemoryWeightStream;
use spm_vectors::{GoldenCase, generate};
use spm_walk::Cursor;

/// Serializes a golden case to `.spm` bytes through the real writer.
fn to_spm(case: &GoldenCase) -> Vec<u8> {
    let descriptors = vec![case.descriptor];
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    let (mut at, mut group) = (0usize, 0usize);
    while let Some(len) = cursor.group_len(&descriptors) {
        let len = len as usize;
        let weights: Vec<Ternary> = case.weights[at..at + len].to_vec();
        writer
            .write_group(case.scales[group], &weights)
            .expect("write group");
        cursor.advance(&descriptors);
        at += len;
        group += 1;
    }
    writer.finish().expect("finish")
}

#[test]
fn the_fabric_matches_the_cpu_reference_bit_for_bit() {
    // Shapes and lane counts chosen to cover weight_lanes below,
    // equal to, and ABOVE the row count -- the last is where the
    // column-major argument is least obvious, because lanes wrap onto
    // the next column.
    let shapes = [(16u32, 5u32, 8u32), (3, 2, 4), (32, 4, 32), (7, 3, 64)];
    let lane_counts = [1usize, 4, 16, 64];
    for (seed, (rows, cols, group)) in shapes.into_iter().enumerate() {
        let case = generate(seed as u64 + 1, rows, cols, group);
        let bytes = to_spm(&case);
        let activations = Activations::broadcast(4, &case.activations);

        let reference =
            run_gemv(MemoryWeightStream::new(bytes.clone()), &activations).expect("reference");

        for weight_lanes in lane_counts {
            let config = FabricConfig {
                weight_lanes,
                ..FabricConfig::unconstrained(4)
            };
            let fabric = run_fabric(
                MemoryWeightStream::new(bytes.clone()),
                &activations,
                &config,
            )
            .expect("fabric");
            assert_eq!(
                fabric.bank, reference.bank,
                "{rows}x{cols} group {group}, {weight_lanes} lanes: not bit-exact"
            );
        }
    }
}

#[test]
fn agreement_holds_across_batch_sizes() {
    let case = generate(21, 16, 5, 8);
    let bytes = to_spm(&case);
    for batch in [1usize, 2, 8, 32] {
        let activations = Activations::broadcast(batch, &case.activations);
        let reference =
            run_gemv(MemoryWeightStream::new(bytes.clone()), &activations).expect("reference");
        let config = FabricConfig {
            weight_lanes: 8,
            ..FabricConfig::unconstrained(batch)
        };
        let fabric = run_fabric(
            MemoryWeightStream::new(bytes.clone()),
            &activations,
            &config,
        )
        .expect("fabric");
        assert_eq!(fabric.bank, reference.bank, "batch {batch}: not bit-exact");
    }
}

#[test]
fn lane_count_changes_cycles_but_never_results() {
    // The separation this model rests on: lanes are a cycle-time
    // choice, not an arithmetic one.
    let case = generate(31, 32, 4, 16);
    let bytes = to_spm(&case);
    let activations = Activations::broadcast(2, &case.activations);
    let one = run_fabric(
        MemoryWeightStream::new(bytes.clone()),
        &activations,
        &FabricConfig {
            weight_lanes: 1,
            ..FabricConfig::unconstrained(2)
        },
    )
    .expect("one lane");
    let many = run_fabric(
        MemoryWeightStream::new(bytes),
        &activations,
        &FabricConfig {
            weight_lanes: 16,
            ..FabricConfig::unconstrained(2)
        },
    )
    .expect("many lanes");
    assert_eq!(one.bank, many.bank, "lane count changed the result");
    assert!(
        many.pipeline.cycles < one.pipeline.cycles,
        "16 lanes ({}) should take fewer cycles than 1 ({})",
        many.pipeline.cycles,
        one.pipeline.cycles
    );
}
