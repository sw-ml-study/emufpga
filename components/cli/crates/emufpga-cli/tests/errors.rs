//! Bad input must fail with a message that names the problem.

use clap::Parser;
use emufpga_cli::{Cli, run};
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("emufpga-err-{name}-{}", std::process::id()))
}

fn cli(args: &[&str]) -> Result<String, String> {
    let parsed = Cli::try_parse_from(args).map_err(|e| e.to_string())?;
    run(parsed).map_err(|e| e.to_string())
}

#[test]
fn a_missing_input_file_names_the_path() {
    let error = cli(&[
        "emufpga",
        "pack",
        "-i",
        "/definitely/not/here.txt",
        "-o",
        "/tmp/unused.spm",
    ])
    .expect_err("must fail");
    assert!(error.contains("/definitely/not/here.txt"), "got {error}");
    assert!(error.contains("cannot read"), "got {error}");
}

#[test]
fn a_malformed_matrix_names_the_line() {
    let input = scratch("ragged.txt");
    std::fs::write(&input, "1 2 3\n4 5\n").expect("write");
    let error = cli(&[
        "emufpga",
        "pack",
        "-i",
        input.to_str().expect("utf8"),
        "-o",
        "/tmp/unused.spm",
    ])
    .expect_err("must fail");
    assert!(error.contains("line 2"), "got {error}");
    std::fs::remove_file(&input).ok();
}

#[test]
fn a_zero_group_size_is_refused_before_any_work() {
    // group_size 0 would divide by zero when counting groups. Caught
    // here rather than deep in the layout crate.
    let input = scratch("zero.txt");
    std::fs::write(&input, "1 2\n").expect("write");
    let error = cli(&[
        "emufpga",
        "pack",
        "-i",
        input.to_str().expect("utf8"),
        "-o",
        "/tmp/unused.spm",
        "-g",
        "0",
    ])
    .expect_err("must fail");
    assert!(
        error.contains("--group-size must be at least 1"),
        "got {error}"
    );
    std::fs::remove_file(&input).ok();
}

#[test]
fn an_unwritable_output_names_the_path() {
    let input = scratch("out.txt");
    std::fs::write(&input, "1 2\n").expect("write");
    let error = cli(&[
        "emufpga",
        "pack",
        "-i",
        input.to_str().expect("utf8"),
        "-o",
        "/definitely/not/a/dir/out.spm",
    ])
    .expect_err("must fail");
    assert!(error.contains("cannot write"), "got {error}");
    std::fs::remove_file(&input).ok();
}
