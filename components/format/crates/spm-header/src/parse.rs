//! Reading a header out of the first bytes of a file.

use crate::model::{Endianness, HEADER_LEN, Header, HeaderError, MAGIC, VERSION_MAJOR};
use spm_bytes::{read_u16, read_u32};

/// Parses the fixed 32-byte header at the start of `src`.
///
/// # Errors
/// Returns [`HeaderError`] if the slice is short, the magic is wrong,
/// the major version is newer than this build, or the endianness
/// discriminant is unknown.
pub fn parse(src: &[u8]) -> Result<Header, HeaderError> {
    if src.len() < HEADER_LEN {
        return Err(HeaderError::TooShort {
            available: src.len(),
        });
    }
    check_magic(src)?;
    let version_major = read_u16(src, 8).ok_or(HeaderError::BadMagic)?;
    check_version(version_major)?;
    Ok(Header {
        version_major,
        version_minor: read_u16(src, 10).ok_or(HeaderError::BadMagic)?,
        endianness: endianness(src[12])?,
        stream_count: read_u32(src, 16).ok_or(HeaderError::BadMagic)?,
    })
}

/// Rejects anything that is not a `.spm` file.
fn check_magic(src: &[u8]) -> Result<(), HeaderError> {
    if src[..MAGIC.len()] == MAGIC {
        Ok(())
    } else {
        Err(HeaderError::BadMagic)
    }
}

/// Refuses a file written by a newer major version.
///
/// Failing loudly matters more than usual here: misparsing a weight
/// stream produces plausible numbers rather than an obvious error.
fn check_version(found: u16) -> Result<(), HeaderError> {
    if found <= VERSION_MAJOR {
        Ok(())
    } else {
        Err(HeaderError::UnsupportedVersion {
            found,
            supported: VERSION_MAJOR,
        })
    }
}

/// Decodes the endianness discriminant.
fn endianness(code: u8) -> Result<Endianness, HeaderError> {
    match code {
        0 => Ok(Endianness::Little),
        code => Err(HeaderError::UnknownEndianness { code }),
    }
}
