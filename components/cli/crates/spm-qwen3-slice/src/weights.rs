use crate::math::matvec;
use spm_gguf::{
    Content, TensorInfo, decode_f32, decode_q6_k, read_tensor_bytes, read_tensor_range,
};
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

pub fn projection_batch(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, String> {
    let weights = load(path, content, name)?;
    inputs
        .iter()
        .map(|input| matvec(&weights, cols, input))
        .collect()
}

pub fn projection_stream(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let info = tensor(content, name)?;
    if info.dtype != 14 || cols % 256 != 0 {
        return Err(format!("tensor {name} is not row-aligned Q6_K"));
    }
    let row_bytes = (cols / 256) * 210;
    let rows = usize::try_from(info.len).map_err(|_| "tensor too large")? / row_bytes;
    let mut output = Vec::with_capacity(rows);
    for first in (0..rows).step_by(256) {
        let count = (rows - first).min(256);
        let offset = u64::try_from(first * row_bytes).map_err(|_| "offset overflow")?;
        let length = u64::try_from(count * row_bytes).map_err(|_| "length overflow")?;
        let bytes = read_tensor_range(path, info, offset, length, length)?;
        output.extend(matvec(&decode_q6_k(&bytes)?, cols, input)?);
    }
    Ok(output)
}
