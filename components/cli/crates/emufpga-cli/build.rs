//! Captures build provenance for `--version`.
//!
//! `sw-checklist` requires Build Host, Build Commit and Build Time in
//! a CLI's version output. They are gathered here rather than at run
//! time because they describe the binary, not the machine running it:
//! a binary copied to another host must still report where it was
//! built.
//!
//! Every field falls back to "unknown" rather than failing the build.
//! A missing `git` or a source tarball outside a repository should not
//! stop the tool from compiling.

use std::process::Command;

fn main() {
    // Re-run when HEAD moves so the recorded commit does not go stale.
    println!("cargo:rerun-if-changed=../../../../.git/HEAD");
    emit("EMUFPGA_BUILD_HOST", &capture("hostname", &[]));
    emit(
        "EMUFPGA_BUILD_COMMIT",
        &capture("git", &["rev-parse", "--short", "HEAD"]),
    );
    emit(
        "EMUFPGA_BUILD_TIME",
        &capture("date", &["-u", "+%Y-%m-%d %H:%M:%S UTC"]),
    );
}

/// Runs `program`, returning trimmed stdout or `"unknown"`.
fn capture(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Exposes `value` to the crate as `env!(name)`.
fn emit(name: &str, value: &str) {
    println!("cargo:rustc-env={name}={value}");
}
