//! The header value and the format constants pinning its layout.

/// File magic. PNG-style, to catch transports that mangle binaries.
pub const MAGIC: [u8; 8] = [0x89, b'S', b'P', b'M', 0x0D, 0x0A, 0x1A, 0x0A];

/// The header occupies a fixed 32 bytes.
pub const HEADER_LEN: usize = 32;

/// Major version this crate writes and is able to read.
///
/// A reader given a larger major version must refuse the file rather
/// than guess at its layout.
pub const VERSION_MAJOR: u16 = 1;

/// Minor version this crate writes. Minor bumps stay readable.
pub const VERSION_MINOR: u16 = 0;

/// Byte order of every multi-byte field in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endianness {
    /// Little-endian. The only order this format defines.
    Little,
}

impl Endianness {
    /// The on-disk discriminant.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Little => 0,
        }
    }
}

/// A parsed `.spm` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Major format version.
    pub version_major: u16,
    /// Minor format version.
    pub version_minor: u16,
    /// Byte order of multi-byte fields.
    pub endianness: Endianness,
    /// Number of entries in the stream directory that follows.
    pub stream_count: u32,
}

impl Header {
    /// A header for a file with `stream_count` streams, at the version
    /// this crate writes.
    #[must_use]
    pub const fn new(stream_count: u32) -> Self {
        Self {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            endianness: Endianness::Little,
            stream_count,
        }
    }
}

/// A header that could not be read.
///
/// Lives beside [`Header`] rather than in its own `error.rs`: this
/// crate's module budget is four, and the header value and its failure
/// modes are one concept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// Fewer than [`HEADER_LEN`] bytes were available.
    TooShort {
        /// Bytes available.
        available: usize,
    },
    /// The magic did not match. Not a `.spm` file, or mangled in
    /// transit by a transport that rewrote line endings.
    BadMagic,
    /// The file was written by a newer major version.
    UnsupportedVersion {
        /// Major version found in the file.
        found: u16,
        /// Largest major version this build understands.
        supported: u16,
    },
    /// The endianness discriminant is not one this format defines.
    UnknownEndianness {
        /// The offending discriminant.
        code: u8,
    },
}

impl core::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort { available } => {
                write!(f, "header truncated: {available} bytes available, need 32")
            }
            Self::BadMagic => write!(f, "bad magic: not a .spm file, or corrupted in transit"),
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "unsupported .spm major version {found}: this build reads up to {supported}"
            ),
            Self::UnknownEndianness { code } => {
                write!(f, "unknown endianness discriminant {code}")
            }
        }
    }
}

impl core::error::Error for HeaderError {}
