//! Stall behaviour, monotonicity, and configurations that are refused.

use fabric_model::{FabricConfig, FabricError, run_fabric};
use spm_activations::Activations;
use spm_codec::Ternary;
use spm_file::SpmWriter;
use spm_stream_mem::MemoryWeightStream;
use spm_vectors::{GoldenCase, generate};
use spm_walk::Cursor;

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

/// A 64x8 case: 512 weights, 128 packed bytes, groups of 64.
fn fixture() -> (Vec<u8>, Vec<f32>) {
    let case = generate(5, 64, 8, 64);
    (to_spm(&case), case.activations)
}

#[test]
fn a_starved_fetch_shows_up_as_stalls() {
    // One byte per cycle against a datapath that eats a whole group
    // at a time: the pipeline should spend most of its cycles waiting.
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(1, &acts);
    let starved = FabricConfig {
        weight_lanes: 64,
        batch_width: 1,
        fifo_bytes: 64,
        fetch_bytes_per_cycle: 1,
        fetch_latency_cycles: 0,
    };
    let run = run_fabric(MemoryWeightStream::new(bytes), &activations, &starved).expect("run");
    let occupancy = run.pipeline.occupancy().expect("occupancy");
    assert!(
        occupancy < 0.2,
        "starved pipeline reported occupancy {occupancy}"
    );
    assert!(run.pipeline.stall_cycles > run.pipeline.cycles / 2);
}

#[test]
fn a_well_fed_fetch_shows_up_as_occupancy() {
    // Wide fetch, narrow datapath: the store keeps up easily and the
    // cycles go into work rather than waiting.
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(32, &acts);
    let fed = FabricConfig {
        weight_lanes: 1,
        batch_width: 1,
        fifo_bytes: 1024,
        fetch_bytes_per_cycle: 256,
        fetch_latency_cycles: 0,
    };
    let run = run_fabric(MemoryWeightStream::new(bytes), &activations, &fed).expect("run");
    let occupancy = run.pipeline.occupancy().expect("occupancy");
    assert!(
        occupancy > 0.9,
        "well-fed pipeline reported occupancy {occupancy}"
    );
}

#[test]
fn more_lanes_never_increase_cycles() {
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(4, &acts);
    let mut previous = u64::MAX;
    for weight_lanes in [1usize, 2, 4, 8, 16, 32, 64] {
        let config = FabricConfig {
            weight_lanes,
            ..FabricConfig::unconstrained(4)
        };
        let run = run_fabric(
            MemoryWeightStream::new(bytes.clone()),
            &activations,
            &config,
        )
        .expect("run");
        assert!(
            run.pipeline.cycles <= previous,
            "{weight_lanes} lanes took {} cycles, more than the narrower config's {previous}",
            run.pipeline.cycles
        );
        previous = run.pipeline.cycles;
    }
}

#[test]
fn a_deeper_fifo_never_increases_stalls() {
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(4, &acts);
    let mut previous = u64::MAX;
    for fifo_bytes in [16usize, 32, 64, 256, 1024] {
        let config = FabricConfig {
            weight_lanes: 8,
            batch_width: 4,
            fifo_bytes,
            fetch_bytes_per_cycle: 16,
            fetch_latency_cycles: 0,
        };
        let run = run_fabric(
            MemoryWeightStream::new(bytes.clone()),
            &activations,
            &config,
        )
        .expect("run");
        assert!(
            run.pipeline.stall_cycles <= previous,
            "fifo {fifo_bytes} stalled {} cycles, more than the shallower config's {previous}",
            run.pipeline.stall_cycles
        );
        previous = run.pipeline.stall_cycles;
    }
}

#[test]
fn degenerate_configurations_are_refused() {
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(1, &acts);
    let cases = [
        (
            FabricConfig {
                weight_lanes: 0,
                ..FabricConfig::unconstrained(1)
            },
            FabricError::MustBePositive {
                field: "weight_lanes",
            },
        ),
        (
            FabricConfig {
                batch_width: 0,
                ..FabricConfig::unconstrained(1)
            },
            FabricError::MustBePositive {
                field: "batch_width",
            },
        ),
        (
            FabricConfig {
                fifo_bytes: 8,
                fetch_bytes_per_cycle: 64,
                ..FabricConfig::unconstrained(1)
            },
            FabricError::FifoSmallerThanFetch {
                fifo_bytes: 8,
                fetch_bytes_per_cycle: 64,
            },
        ),
    ];
    for (config, expected) in cases {
        let error = run_fabric(
            MemoryWeightStream::new(bytes.clone()),
            &activations,
            &config,
        )
        .expect_err("must be refused");
        assert_eq!(error, expected);
    }
}

#[test]
fn startup_latency_is_paid_once() {
    // A scan is one continuous stream. A real store pays latency on
    // every discontinuity; the whole point of the architecture is
    // that there are none.
    let (bytes, acts) = fixture();
    let activations = Activations::broadcast(1, &acts);
    let base = FabricConfig::unconstrained(1);
    let delayed = FabricConfig {
        fetch_latency_cycles: 1000,
        ..base
    };
    let a = run_fabric(MemoryWeightStream::new(bytes.clone()), &activations, &base).expect("a");
    let b = run_fabric(MemoryWeightStream::new(bytes), &activations, &delayed).expect("b");
    assert_eq!(b.pipeline.cycles - a.pipeline.cycles, 1000);
}
