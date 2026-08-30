//! Batch reuse, backend equivalence, and the first end-to-end check
//! that the Ps accounting is actually wired to a running engine.

mod support;

use spm_gemv_ref::{Activations, run_gemv};
use spm_stream_file::FileWeightStream;
use spm_stream_mem::MemoryWeightStream;
use spm_vectors::generate;
use std::io::Write;
use support::to_spm;

#[test]
fn every_lane_agrees_when_every_lane_holds_the_same_activations() {
    // Batch invariance. If lanes diverge under identical inputs, the
    // batch dimension is indexed wrong -- which would otherwise show
    // up much later as a quietly incorrect throughput number.
    let case = generate(21, 16, 5, 8);
    let bytes = to_spm(&case);
    let single = run_gemv(
        MemoryWeightStream::new(bytes.clone()),
        &Activations::broadcast(1, &case.activations),
    )
    .expect("batch 1");
    let batched = run_gemv(
        MemoryWeightStream::new(bytes),
        &Activations::broadcast(8, &case.activations),
    )
    .expect("batch 8");

    for lane in 0..8 {
        assert_eq!(
            batched.bank.lane(lane),
            single.bank.lane(0),
            "lane {lane} diverged from the batch-1 result"
        );
    }
}

#[test]
fn scan_productivity_equals_the_batch_size() {
    // Driven by a real scan rather than hand-built counters, so this
    // is the first check that the metric is wired to the engine at
    // all and not merely arithmetically correct in isolation.
    let case = generate(23, 16, 4, 8);
    let bytes = to_spm(&case);
    for lanes in [1usize, 2, 4, 8, 16, 32] {
        let outcome = run_gemv(
            MemoryWeightStream::new(bytes.clone()),
            &Activations::broadcast(lanes, &case.activations),
        )
        .expect("run gemv");
        let ps = outcome.metrics.scan_productivity().expect("Ps");
        let want = f64::from(u32::try_from(lanes).expect("lane count fits u32"));
        assert!(
            (ps - want).abs() < 1e-9,
            "batch {lanes}: Ps was {ps}, expected {want}"
        );
        // The bytes read off the store do NOT change with batch size.
        // That is the entire economic argument, so assert it rather
        // than trusting Ps alone.
        assert_eq!(
            outcome.metrics.parameter_bytes_read,
            (16 * 4u64).div_ceil(4),
            "batch {lanes} changed the bytes read"
        );
        assert_eq!(outcome.metrics.weights_decoded, 64);
        assert_eq!(outcome.metrics.weight_applications, 64 * lanes as u64);
    }
}

#[test]
fn residency_stays_far_below_one() {
    // Rp is the claim that the model need not be resident. One group
    // buffer against the whole matrix.
    let case = generate(29, 64, 64, 64);
    let outcome = run_gemv(
        MemoryWeightStream::new(to_spm(&case)),
        &Activations::broadcast(1, &case.activations),
    )
    .expect("run gemv");
    let rp = outcome.metrics.residency().expect("Rp");
    // 16 bytes of group buffer against 1024 bytes of packed weights.
    assert!((rp - 16.0 / 1024.0).abs() < 1e-9, "Rp was {rp}");
    assert!(rp < 0.02, "Rp {rp} is not far below 1");
}

#[test]
fn the_file_backend_produces_identical_accumulators() {
    // Whatever is measured against one backend has to predict the
    // other, or the reference implementation is not a reference.
    let case = generate(31, 16, 5, 16);
    let bytes = to_spm(&case);
    let path = std::env::temp_dir().join(format!("emufpga-gemv-{}.spm", std::process::id()));
    std::fs::File::create(&path)
        .expect("create")
        .write_all(&bytes)
        .expect("write");

    let from_memory = run_gemv(
        MemoryWeightStream::new(bytes),
        &Activations::broadcast(4, &case.activations),
    )
    .expect("memory");
    // A tiny capacity forces refills mid-group, exercising the
    // short-read path underneath the group reader.
    let from_file = run_gemv(
        FileWeightStream::with_capacity(&path, 7).expect("open"),
        &Activations::broadcast(4, &case.activations),
    )
    .expect("file");

    assert_eq!(from_file.bank, from_memory.bank);
    assert_eq!(
        from_file.metrics.weights_decoded,
        from_memory.metrics.weights_decoded
    );
    std::fs::remove_file(&path).ok();
}
