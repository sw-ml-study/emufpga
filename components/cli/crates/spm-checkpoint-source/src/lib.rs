//! Framework checkpoint metadata and raw tensor access.

mod model;
mod parse;
mod safetensors;

pub use model::{DType, TensorSource, read_tensor};
pub use parse::open;
