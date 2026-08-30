//! The ternary weight value and its two-bit wire code.

use crate::error::CodecError;

/// Bit 0 of a code: the weight is nonzero.
///
/// In hardware this line IS the accumulator enable. Exported as a
/// constant rather than wrapped in a method because the fabric wires
/// it directly.
pub const NONZERO_BIT: u8 = 0b01;
/// Bit 1 of a code: the weight is negative.
///
/// In hardware this line selects subtract over add.
pub const NEGATIVE_BIT: u8 = 0b10;
/// Mask selecting one weight's code out of a packed byte.
pub(crate) const CODE_MASK: u8 = 0b11;

/// A single ternary weight: `-1`, `0` or `+1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    /// Contributes nothing to the accumulator.
    Zero,
    /// Adds the activation to the accumulator.
    Plus,
    /// Subtracts the activation from the accumulator.
    Minus,
}

impl Ternary {
    /// The two-bit wire code for this weight.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::Plus => NONZERO_BIT,
            Self::Minus => NONZERO_BIT | NEGATIVE_BIT,
        }
    }

    /// Decodes a two-bit wire code.
    ///
    /// # Errors
    /// Returns [`CodecError::ReservedCode`] for `0b10`, the invalid
    /// "negative zero" pattern that this encoding never emits.
    pub const fn from_code(code: u8) -> Result<Self, CodecError> {
        match code & CODE_MASK {
            0 => Ok(Self::Zero),
            NONZERO_BIT => Ok(Self::Plus),
            c if c == NONZERO_BIT | NEGATIVE_BIT => Ok(Self::Minus),
            c => Err(CodecError::ReservedCode { code: c }),
        }
    }

    /// The numeric value: `-1`, `0` or `+1`.
    #[must_use]
    pub const fn value(self) -> i32 {
        match self {
            Self::Zero => 0,
            Self::Plus => 1,
            Self::Minus => -1,
        }
    }

    /// Builds a weight from a numeric value.
    ///
    /// # Errors
    /// Returns [`CodecError::NotTernary`] for anything outside
    /// `-1..=1`. Quantization happens before packing; this constructor
    /// deliberately refuses to round for you.
    pub const fn from_value(value: i32) -> Result<Self, CodecError> {
        match value {
            0 => Ok(Self::Zero),
            1 => Ok(Self::Plus),
            -1 => Ok(Self::Minus),
            v => Err(CodecError::NotTernary { value: v }),
        }
    }
}
