use crate::math::matvec;
use spm_gguf::{Content, TensorInfo, decode_f32, decode_q6_k, read_tensor_bytes};
use std::path::Path;

const MAX_TENSOR_BYTES: u64 = 64 * 1024 * 1024;

pub fn tensor<'a>(content: &'a Content, name: &str) -> Result<&'a TensorInfo, String> {
    content
        .tensors
        .iter()
        .find(|item| item.name == name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

pub fn load(path: &Path, content: &Content, name: &str) -> Result<Vec<f32>, String> {
    let info = tensor(content, name)?;
    let bytes = read_tensor_bytes(path, info, MAX_TENSOR_BYTES)?;
    match info.dtype {
        0 => decode_f32(&bytes),
        14 => decode_q6_k(&bytes),
        other => Err(format!("tensor {name} has unsupported dtype {other}")),
    }
}

pub fn projection(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    matvec(&load(path, content, name)?, cols, input)
}
