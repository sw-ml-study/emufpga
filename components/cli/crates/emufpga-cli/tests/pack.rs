//! `pack` end to end: the packer is only correct if the engine
//! reproduces the reference from its output.

use clap::Parser;
use emufpga_cli::{Cli, run};
use spm_gemv_ref::{Activations, run_gemv};
use spm_numeric::{DenseTernary, max_abs_error, reference_gemv};
use spm_quantize::{parse_matrix, quantize, write_spm};
use spm_stream_mem::MemoryWeightStream;
use std::path::PathBuf;

/// A scratch path unique to this process and `name`.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("emufpga-pack-{name}-{}", std::process::id()))
}

/// Runs the CLI exactly as the binary would.
fn cli(args: &[&str]) -> Result<String, String> {
    let parsed = Cli::try_parse_from(args).map_err(|e| e.to_string())?;
    run(parsed).map_err(|e| e.to_string())
}

/// A matrix with a spread of magnitudes and signs, including zeros.
const MATRIX: &str = "\
 1.0  -3.0   0.0   4.0   0.5
-2.5   0.0   1.25 -0.75  2.0
 0.0   6.0  -1.0   0.25 -3.5
 3.0   1.5   0.0  -2.0   0.0
";

#[test]
fn the_engine_reproduces_the_reference_from_the_packed_file() {
    // The whole point of the subcommand: bytes it writes must drive
    // the streaming engine to the same answer the naive reference
    // computes from the same quantized weights.
    let input = scratch("engine.txt");
    let output = scratch("engine.spm");
    std::fs::write(&input, MATRIX).expect("write matrix");

    cli(&[
        "emufpga",
        "pack",
        "--input",
        input.to_str().expect("utf8"),
        "--output",
        output.to_str().expect("utf8"),
        "--group-size",
        "4",
    ])
    .expect("pack");

    let quantized = quantize(&parse_matrix(MATRIX).expect("parse"), 4);
    let activations = [0.5f32, -1.5, 2.0, 0.25, -0.75];
    let expected = reference_gemv(
        &DenseTernary {
            rows: 4,
            cols: 5,
            group_size: 4,
            weights: quantized.weights.clone(),
            scales: quantized.scales.clone(),
        },
        &activations,
    );

    let bytes = std::fs::read(&output).expect("read spm");
    let outcome = run_gemv(
        MemoryWeightStream::new(bytes),
        &Activations::broadcast(1, &activations),
    )
    .expect("gemv");

    let error = max_abs_error(outcome.bank.lane(0), &expected);
    assert!(
        error <= 1e-4,
        "max abs error {error}\ngot {:?}",
        outcome.bank.lane(0)
    );
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

#[test]
fn the_cli_writes_exactly_what_the_library_writes() {
    // Guards against the CLI drifting from spm-quantize -- a second
    // packing path would be a second thing to keep correct.
    let input = scratch("drift.txt");
    let output = scratch("drift.spm");
    std::fs::write(&input, MATRIX).expect("write matrix");
    cli(&[
        "emufpga",
        "pack",
        "-i",
        input.to_str().expect("utf8"),
        "-o",
        output.to_str().expect("utf8"),
        "-g",
        "8",
    ])
    .expect("pack");

    let direct = write_spm(&quantize(&parse_matrix(MATRIX).expect("parse"), 8)).expect("write");
    assert_eq!(std::fs::read(&output).expect("read"), direct);
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

#[test]
fn the_default_group_size_is_documented_and_applied() {
    // 64, per the --help text. Asserted so the doc and the behaviour
    // cannot drift apart.
    let input = scratch("default.txt");
    let output = scratch("default.spm");
    std::fs::write(&input, MATRIX).expect("write matrix");
    let summary = cli(&[
        "emufpga",
        "pack",
        "-i",
        input.to_str().expect("utf8"),
        "-o",
        output.to_str().expect("utf8"),
    ])
    .expect("pack");
    assert!(summary.contains("group size 64"), "got {summary}");
    // 4x5 = 20 weights, one group of 64 covers them all.
    assert!(summary.contains("1 groups"), "got {summary}");
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}
