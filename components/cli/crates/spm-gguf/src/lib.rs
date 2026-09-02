//! Bounded, non-executing GGUF metadata and tensor-range reader.
mod model;
mod parse;
mod tensor;
mod wire;
pub use model::{Content, TensorInfo};
pub use parse::read;
pub use tensor::{decode_f32, decode_q6_k, read_tensor_bytes, read_tensor_range};
pub const MAX_METADATA: u64 = 1_000_000;
pub const MAX_TENSORS: u64 = 1_000_000;
pub const MAX_STRING: u64 = 16 * 1024 * 1024;
pub const MAX_ARRAY: u64 = 2_000_000;
pub const MAX_DIMS: u32 = 8;
pub const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
