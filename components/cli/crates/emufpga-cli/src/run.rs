//! Dispatch, and the failure type the subcommands share.

use crate::args::{Cli, Command};
use crate::commands::{bench, pack, sim};
use std::fmt;

/// A subcommand that could not complete its work.
///
/// Carries a message rather than a typed cause: everything the CLI can
/// fail at ends up as one line on stderr and exit code 1, and a deep
/// error hierarchy for that would be structure without a consumer.
#[derive(Debug)]
pub struct Failure(String);

impl Failure {
    /// A failure with `message`.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Failure {}

/// Runs a parsed command, returning the line to print on success.
///
/// Adding a subcommand is an arm here plus a function in
/// `commands.rs`, which is why `fit` will not need this restructured.
///
/// # Errors
/// Returns [`Failure`] if the subcommand's work fails.
pub fn run(cli: Cli) -> Result<String, Failure> {
    match cli.command {
        Command::Pack {
            input,
            output,
            group_size,
        } => pack(&input, &output, group_size),
        Command::Bench {
            input,
            batch,
            repeat,
        } => bench(&input, &batch, repeat),
        Command::Sim(args) => sim(&args.input, &args.config(), args.batch),
    }
}
