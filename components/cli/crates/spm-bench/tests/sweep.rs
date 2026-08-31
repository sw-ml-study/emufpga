//! The sweep must report what the engine computes, and must not
//! invent a crossover it did not observe.

use spm_bench::{Backend, Crossover, run_sweep, timer_overhead};
use spm_bench_report::render;
use spm_quantize::{parse_matrix, quantize, write_spm};
use std::path::PathBuf;

/// Writes a small .spm to a scratch path and returns it.
fn fixture(name: &str, group_size: u32) -> PathBuf {
    let matrix = (0..16)
        .map(|r| {
            (0..8)
                .map(|c| format!("{:.2}", f64::from(r * 8 + c).mul_add(0.37, -3.0)))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let quantized = quantize(&parse_matrix(&matrix).expect("parse"), group_size);
    let path =
        std::env::temp_dir().join(format!("emufpga-bench-{name}-{}.spm", std::process::id()));
    std::fs::write(&path, write_spm(&quantized).expect("write")).expect("write file");
    path
}

#[test]
fn the_sweep_reports_the_metrics_the_engine_computes() {
    // Ps and Rp are pinned by step 004's tests against the engine
    // directly. If bench reported different numbers, it would be
    // measuring something other than the engine.
    let path = fixture("metrics", 16);
    let sweep = run_sweep(&path, &[1, 2, 4], 2).expect("sweep");
    assert_eq!(sweep.rows.len(), 6, "two backends x three batch sizes");
    for row in &sweep.rows {
        let ps = row.best.scan_productivity().expect("Ps");
        let want = f64::from(u32::try_from(row.batch).expect("fits"));
        assert!((ps - want).abs() < 1e-9, "batch {}: Ps {ps}", row.batch);
        // 16x8 = 128 weights = 32 packed bytes, whatever the batch.
        assert_eq!(row.best.parameter_bytes_read, 32, "batch {}", row.batch);
        assert_eq!(row.best.weights_decoded, 128);
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn both_backends_read_the_same_bytes() {
    // Storage traffic is a property of the file, not of how it was
    // opened. If these diverged, eta would not be comparable across
    // backends and the report would be meaningless.
    let path = fixture("backends", 16);
    let sweep = run_sweep(&path, &[1, 8], 2).expect("sweep");
    let memory: Vec<u64> = sweep
        .rows
        .iter()
        .filter(|r| r.backend == Backend::Memory)
        .map(|r| r.best.parameter_bytes_read)
        .collect();
    let file: Vec<u64> = sweep
        .rows
        .iter()
        .filter(|r| r.backend == Backend::File)
        .map(|r| r.best.parameter_bytes_read)
        .collect();
    assert_eq!(memory, file);
    std::fs::remove_file(&path).ok();
}

#[test]
fn an_unmeasured_crossover_is_reported_as_unmeasured() {
    // The CPU reference is compute-bound from batch 1, so no crossing
    // is observed. Reporting "crossover at batch 1" would claim a
    // measurement that was never made -- the distinction this enum
    // exists to keep.
    let path = fixture("crossover", 16);
    let sweep = run_sweep(&path, &[1, 2, 4], 2).expect("sweep");
    assert_eq!(
        sweep.crossover(Backend::File),
        Crossover::AlreadyBelow { smallest: 1 }
    );
    let text = render(&sweep);
    assert!(text.contains("NOT MEASURED"), "{text}");
    assert!(text.contains("IO is NOT overlapped"), "{text}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn timer_overhead_is_measured_and_nonzero() {
    // Charged to every scale group, so a report that ignored it could
    // present timing noise as a storage/compute split.
    let overhead = timer_overhead();
    assert!(overhead.as_nanos() > 0, "timestamp pair took no time");
    assert!(
        overhead.as_micros() < 10,
        "implausible overhead {overhead:?}"
    );
}

#[test]
fn a_missing_file_fails_rather_than_reporting_zeros() {
    let missing = std::env::temp_dir().join("emufpga-bench-nonexistent.spm");
    assert!(run_sweep(&missing, &[1], 1).is_err());
}
