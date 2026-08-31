//! The `emufpga` command line.
//!
//! Split into a library and a thin binary so integration tests call
//! [`run`] directly instead of shelling out. Shelling out would test
//! the same code plus a process boundary, and would make a failure
//! harder to read.
//!
//! Exit codes are part of the interface:
//!
//! | code | meaning |
//! |------|---------|
//! | 0    | success |
//! | 1    | the work failed (unreadable input, malformed matrix) |
//! | 2    | the command line itself was wrong (clap's convention) |
//!
//! Saga 1 ships `pack` and `bench`; `fit` arrives in step 008. Each
//! subcommand is one function in `commands.rs` returning the text to
//! print, so adding one is an enum arm, a function, and a match arm --
//! never a restructure.

mod args;
mod commands;
mod run;

pub use args::{Cli, Command, SimArgs};
pub use run::{Failure, run};
