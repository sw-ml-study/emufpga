//! The command line surface `sw-checklist` validates, asserted here so
//! a regression shows up in `cargo test` rather than only in the
//! standards check.

use clap::{CommandFactory, Parser};
use emufpga_cli::Cli;

#[test]
fn long_help_is_longer_than_short_help() {
    let short = Cli::command().render_help().to_string();
    let long = Cli::command().render_long_help().to_string();
    assert!(
        long.len() > short.len(),
        "--help ({}) must be longer than -h ({})",
        long.len(),
        short.len()
    );
}

#[test]
fn long_help_carries_the_agent_section() {
    let long = Cli::command().render_long_help().to_string();
    assert!(long.contains("AI CODING AGENT INSTRUCTIONS"), "{long}");
    // The two invariants an agent most needs to not break.
    assert!(long.contains("never seeks"), "{long}");
    assert!(long.contains("PHYSICAL EXECUTION LAYOUT"), "{long}");
}

#[test]
fn version_carries_every_field_the_standard_requires() {
    let version = Cli::command().render_version();
    for field in [
        "Copyright",
        "License",
        "Repository",
        "Build Host",
        "Build Commit",
        "Build Time",
    ] {
        assert!(
            version.contains(field),
            "version missing {field}:\n{version}"
        );
    }
}

#[test]
fn a_usage_error_is_distinguishable_from_a_work_failure() {
    // clap reports usage errors itself and the binary exits 2; work
    // failures come back as Err from run() and exit 1. Keeping the two
    // apart is what lets a script tell "I typed it wrong" from "the
    // input was bad".
    let usage = Cli::try_parse_from(["emufpga", "pack"]).expect_err("missing args");
    assert_eq!(usage.exit_code(), 2);
    assert!(Cli::try_parse_from(["emufpga", "--version"]).is_err());
}
