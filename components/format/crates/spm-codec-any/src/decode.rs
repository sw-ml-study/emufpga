//! The dispatch.

use core::fmt;
use spm_layout::Encoding;

/// Why a group could not be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The source was shorter than the encoding needs. Carries the
    /// byte count required.
    Short(usize),
    /// This build has no dense decoder for that profile.
    ///
    /// Ternary lands here: its weights are two-bit codes that mean
    /// nothing without their group scale and an accumulator that adds
    /// and subtracts rather than multiplies, so it needs a
    /// scale-aware engine rather than a wider buffer.
    Unsupported(Encoding),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short(needed) => write!(f, "group needs {needed} bytes"),
            Self::Unsupported(encoding) => {
                write!(f, "{encoding:?} needs a scale-aware engine")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decodes one group into `dst`, choosing the codec by `encoding`.
///
/// # Errors
/// Returns [`DecodeError`] if `src` is too short or the profile has no
/// dense decoder.
pub fn decode_into(encoding: Encoding, src: &[u8], dst: &mut [f32]) -> Result<(), DecodeError> {
    match encoding {
        Encoding::F32 => spm_codec_dense::decode_into(src, dst).map_err(DecodeError::Short),
        Encoding::Bf16 => spm_codec_bf16::decode_into(src, dst).map_err(DecodeError::Short),
        Encoding::Ternary2F32I32 => Err(DecodeError::Unsupported(encoding)),
    }
}
