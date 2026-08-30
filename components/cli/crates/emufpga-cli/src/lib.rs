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
//! Saga 1 ships `pack`. `bench` arrives in step 006 and `fit` in step
//! 008, so [`Command`] is an enum with one arm today and dispatch in
//! [`run`] is a match -- adding a subcommand is a new arm and a new
//! module, not a rewrite.

mod args;
mod pack;
mod run;

pub use args::{Cli, Command};
pub use run::{Failure, run};
