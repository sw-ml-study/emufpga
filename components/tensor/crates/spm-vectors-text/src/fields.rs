//! Field parsers and the error they raise.

use spm_codec::Ternary;
use std::fmt;

/// A golden case file that could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A required header line was missing or malformed.
    BadHeader {
        /// The line at fault.
        line: String,
    },
    /// A weight was not one of `-`, `0`, `+`.
    BadWeight {
        /// The offending character.
        found: char,
    },
    /// A numeric field did not parse.
    BadNumber {
        /// The offending token.
        found: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadHeader { line } => write!(f, "malformed header line: {line}"),
            Self::BadWeight { found } => write!(f, "bad weight character {found:?}"),
            Self::BadNumber { found } => write!(f, "bad number {found:?}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parses the three integers of a `shape` line.
///
/// # Errors
/// Returns [`ParseError`] if a field is missing or not an integer.
pub(crate) fn shape(rest: &str) -> Result<[u32; 3], ParseError> {
    let mut out = [0u32; 3];
    let mut tokens = rest.split_whitespace();
    for slot in &mut out {
        let token = tokens.next().ok_or_else(|| ParseError::BadHeader {
            line: format!("shape {rest}"),
        })?;
        *slot = token.parse().map_err(|_| ParseError::BadNumber {
            found: token.trim().into(),
        })?;
    }
    Ok(out)
}

/// Parses a whitespace-separated list of floats.
///
/// # Errors
/// Returns [`ParseError::BadNumber`] for any token that is not a
/// float.
pub(crate) fn numbers(rest: &str) -> Result<Vec<f32>, ParseError> {
    rest.split_whitespace()
        .map(|token| {
            token.parse().map_err(|_| ParseError::BadNumber {
                found: token.trim().into(),
            })
        })
        .collect()
}

/// Parses a run of `-`, `0`, `+` weight symbols.
///
/// # Errors
/// Returns [`ParseError::BadWeight`] for any other character.
pub(crate) fn weights(rest: &str) -> Result<Vec<Ternary>, ParseError> {
    rest.trim()
        .chars()
        .map(|found| match found {
            '0' => Ok(Ternary::Zero),
            '+' => Ok(Ternary::Plus),
            '-' => Ok(Ternary::Minus),
            found => Err(ParseError::BadWeight { found }),
        })
        .collect()
}
