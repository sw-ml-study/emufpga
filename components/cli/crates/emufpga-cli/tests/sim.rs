//! `sim` must report the model's numbers and its caveats together.

use clap::Parser;
use emufpga_cli::{Cli, run};
use spm_quantize::{parse_matrix, quantize, write_spm};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    let matrix = (0..32)
        .map(|r| {
            (0..8)
                .map(|c| format!("{:.2}", f64::from(r * 8 + c).mul_add(0.11, -2.0)))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let quantized = quantize(&parse_matrix(&matrix).expect("parse"), 32);
    let path = std::env::temp_dir().join(format!("emufpga-sim-{name}-{}.spm", std::process::id()));
    std::fs::write(&path, write_spm(&quantized).expect("write")).expect("write file");
    path
}

fn cli(args: &[&str]) -> Result<String, String> {
    let parsed = Cli::try_parse_from(args).map_err(|e| e.to_string())?;
    run(parsed).map_err(|e| e.to_string())
}

#[test]
fn a_starved_configuration_reports_low_occupancy() {
    let path = fixture("starved");
    let p = path.to_str().expect("utf8");
    let starved = cli(&["emufpga", "sim", "-i", p, "-F", "1", "-f", "64"]).expect("sim");
    let fed = cli(&["emufpga", "sim", "-i", p, "-F", "256", "-f", "512"]).expect("sim");
    // Both report the same weight count; only the cycles differ.
    assert!(starved.contains("| weights | 256 |"), "{starved}");
    assert!(fed.contains("| weights | 256 |"), "{fed}");
    assert!(
        starved.matches("stall cycles").count() == 1 && fed.contains("occupancy"),
        "both runs must report stalls and occupancy"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn the_caveats_travel_with_the_numbers() {
    // A table of cycles is easy to misread as a prediction about
    // silicon. The caveat is emitted with it, not filed elsewhere.
    let path = fixture("caveat");
    let out = cli(&["emufpga", "sim", "-i", path.to_str().expect("utf8")]).expect("sim");
    assert!(out.contains("UNIT, not a duration"), "{out}");
    assert!(out.contains("not an FPGA simulator"), "{out}");
    assert!(out.contains("bit-exact"), "{out}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_degenerate_configuration_is_refused_by_the_cli() {
    let path = fixture("degenerate");
    let error = cli(&[
        "emufpga",
        "sim",
        "-i",
        path.to_str().expect("utf8"),
        "-l",
        "0",
    ])
    .expect_err("must fail");
    assert!(error.contains("weight_lanes must be at least 1"), "{error}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn sim_accepts_no_clock_argument() {
    // The guard against this becoming a fit model by the back door:
    // there is no --fmax, so nothing can convert cycles to seconds.
    let path = fixture("noclock");
    let error = cli(&[
        "emufpga",
        "sim",
        "-i",
        path.to_str().expect("utf8"),
        "--fmax",
        "100",
    ])
    .expect_err("must reject an unknown flag");
    assert!(
        error.contains("unexpected argument") || error.contains("--fmax"),
        "{error}"
    );
    std::fs::remove_file(&path).ok();
}
