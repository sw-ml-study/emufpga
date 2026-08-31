//! A number, and where it came from -- or an honest gap.

use core::fmt;

/// Where a figure was read from.
///
/// Plain public fields and no `Display`: there is no consumer that
/// needs one yet, and step 008's report will want to lay these out
/// its own way rather than accept a format decided here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    /// Document title or identifier.
    pub document: &'static str,
    /// Where it was read.
    pub url: &'static str,
    /// ISO date it was retrieved, so a stale figure is visible as
    /// stale rather than merely wrong.
    pub retrieved: &'static str,
}

/// A device figure that may not have been sourceable.
///
/// Deliberately not `Option<u32>`. An `Option` invites `unwrap_or(0)`,
/// and a zero LUT4 count that reads as a number is exactly the failure
/// this type exists to prevent. Getting a value out requires
/// acknowledging it might not be there, and an [`Figure::Unknown`]
/// carries a note saying what was tried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Figure {
    /// A value read from a named document.
    Known {
        /// The value.
        value: u64,
        /// Where it came from.
        source: Source,
    },
    /// Not sourceable. The note says what was looked at.
    Unknown {
        /// What was tried, so the next attempt does not repeat it.
        note: &'static str,
    },
}

impl Figure {
    /// The value, if it was sourced.
    #[must_use]
    pub const fn value(self) -> Option<u64> {
        match self {
            Self::Known { value, .. } => Some(value),
            Self::Unknown { .. } => None,
        }
    }

    /// Whether this figure was sourced.
    #[must_use]
    pub const fn is_known(self) -> bool {
        matches!(self, Self::Known { .. })
    }

    /// The document behind this figure, if any.
    #[must_use]
    pub const fn source(self) -> Option<Source> {
        match self {
            Self::Known { source, .. } => Some(source),
            Self::Unknown { .. } => None,
        }
    }
}

impl fmt::Display for Figure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known { value, .. } => write!(f, "{value}"),
            Self::Unknown { note } => write!(f, "unknown ({note})"),
        }
    }
}
