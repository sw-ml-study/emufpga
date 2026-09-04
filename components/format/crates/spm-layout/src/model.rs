//! The operation descriptor and the datapath profile it names.

/// Each descriptor occupies a fixed 32 bytes.
pub const DESCRIPTOR_LEN: usize = 32;

/// The datapath profile: weight encoding, scale type and accumulator
/// width, named as one value.
///
/// These three co-vary in practice -- an FPGA is built for one
/// combination, not for a matrix of independent choices -- so the
/// format names the combination rather than three orthogonal fields.
/// Unknown discriminants are rejected rather than ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Two-bit ternary weights, `f32` group scales, `i32`
    /// accumulators. Saga 1's profile.
    Ternary2F32I32,
    /// Dense `f32` weights, four bytes each, no packing.
    ///
    /// The group scale is **inert** for this profile: the weights
    /// carry their own magnitude, so the writer emits `1.0` and
    /// readers ignore it. The group structure is kept anyway, because
    /// it is what makes the stream self-describing.
    F32,
    /// `bfloat16` weights, two bytes each, no packing.
    ///
    /// The top 16 bits of an `f32`: same 8-bit exponent, 8 fewer
    /// mantissa bits. Chosen over `f16` because it is what checkpoints
    /// are actually stored in, so importing one is a truncation rather
    /// than a range conversion, and nothing can overflow.
    ///
    /// The group scale is **inert**, as for [`Self::F32`]: the weights
    /// carry their own magnitude.
    Bf16,
    /// GGML `Q6_K` blocks in source row-major order: 256 weights in 210 bytes.
    ///
    /// The quantization scale is internal to each `Q6_K` block. The outer `.spm`
    /// group scale is therefore inert, as for the dense profiles.
    Q6K,
}

impl Encoding {
    /// The on-disk discriminant.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Ternary2F32I32 => 1,
            Self::F32 => 2,
            Self::Bf16 => 3,
            Self::Q6K => 4,
        }
    }

    /// Bytes a group of `count` weights occupies on the wire.
    ///
    /// The reason this method exists. Every group sizing in the tree
    /// used to call `spm_codec::packed_len`, which hardcodes ternary's
    /// two bits per weight -- so the `Encoding` discriminant existed
    /// to allow a second encoding while nothing consulted it when
    /// computing bytes. Ask the encoding instead.
    #[must_use]
    pub const fn bytes_for(self, count: usize) -> usize {
        match self {
            // Four weights to a byte, groups byte-aligned.
            Self::Ternary2F32I32 => count.div_ceil(4),
            Self::F32 => count * 4,
            Self::Bf16 => count * 2,
            Self::Q6K => count.div_ceil(256) * 210,
        }
    }

    /// Decodes a manifest `dtype` name.
    ///
    /// The extractor writes the dtype of the blob it produced, and
    /// every stage downstream must agree with it: a bf16 blob framed
    /// as f32 is read as garbage with no error anywhere. Deliberately
    /// strict -- an unrecognised name is refused rather than guessed
    /// at, because the guess would be silent.
    #[must_use]
    pub fn from_dtype(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(Self::F32),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Decodes an on-disk discriminant.
    ///
    /// # Errors
    /// Returns [`LayoutError::UnknownEncoding`] for any value this
    /// build does not implement.
    pub const fn from_code(code: u8) -> Result<Self, LayoutError> {
        match code {
            1 => Ok(Self::Ternary2F32I32),
            2 => Ok(Self::F32),
            3 => Ok(Self::Bf16),
            4 => Ok(Self::Q6K),
            code => Err(LayoutError::UnknownEncoding { code }),
        }
    }
}

/// One streamed operation: a matrix and how to consume it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpDescriptor {
    /// Output count, M. The number of accumulators the engine needs.
    pub rows: u32,
    /// Input count, N. The number of activations held resident.
    pub cols: u32,
    /// Weights per scale group, in stream order.
    pub group_size: u32,
    /// Weight encoding, scale type and accumulator width.
    pub encoding: Encoding,
    /// Parallel weight lanes the stream is striped across.
    pub lane_count: u16,
}

/// A descriptor that could not be read, or does not describe a
/// consumable stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// Fewer than [`DESCRIPTOR_LEN`] bytes were available.
    TooShort {
        /// Bytes available.
        available: usize,
    },
    /// The encoding discriminant is not one this build implements.
    UnknownEncoding {
        /// The offending discriminant.
        code: u8,
    },
    /// `group_size` was zero, which would divide by zero when
    /// counting groups.
    ZeroGroupSize,
}
