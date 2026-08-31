//! Decoding a group by its declared encoding.
//!
//! One place that knows every profile, so no reader has to. Before
//! this existed, `spm_linear` called the f32 codec unconditionally
//! and a bf16 stream would have been read as f32 -- plausible garbage
//! with no error anywhere, which is the failure mode
//! docs/postmortem-1.md keeps finding.
//!
//! Deliberately decode-only. Encoding happens in the importer, which
//! knows which profile it is writing and can call the codec directly;
//! a symmetric `encode_into` here would invite callers to pick an
//! encoding at the point of writing rather than at the point of
//! deciding.

mod decode;

pub use decode::{DecodeError, decode_into};
