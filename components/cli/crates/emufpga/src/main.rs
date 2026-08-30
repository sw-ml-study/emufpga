//! The `emufpga` binary.
//!
//! Deliberately thin: parse, dispatch, print, choose an exit code.
//! Everything worth testing lives in `emufpga-cli`.

use clap::Parser;
use emufpga_cli::{Cli, run};

fn main() -> std::process::ExitCode {
    // clap exits with code 2 on a usage error before returning here.
    let cli = Cli::parse();
    match run(cli) {
        Ok(summary) => {
            println!("{summary}");
            std::process::ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("emufpga: {failure}");
            std::process::ExitCode::FAILURE
        }
    }
}
