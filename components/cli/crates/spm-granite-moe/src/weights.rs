use crate::math::matvec;
use spm_gguf::{
    Content, TensorInfo, decode_f32, decode_q6_k, read_tensor_bytes, read_tensor_range,
};
use std::path::Path;

const MAX_BYTES: u64 = 16 * 1024 * 1024;

pub fn tensor<'a>(content: &'a Content, name: &str) -> Result<&'a TensorInfo, String> {
    content
        .tensors
        .iter()
        .find(|item| item.name == name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

pub fn load(path: &Path, content: &Content, name: &str) -> Result<Vec<f32>, String> {
    let info = tensor(content, name)?;
    let bytes = read_tensor_bytes(path, info, MAX_BYTES)?;
    match info.dtype {
        0 => decode_f32(&bytes),
        14 => decode_q6_k(&bytes),
        kind => Err(format!("tensor {name} has unsupported dtype {kind}")),
    }
}

pub fn project_batch(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, String> {
    let values = load(path, content, name)?;
    inputs
        .iter()
        .map(|input| matvec(&values, cols, input))
        .collect()
}

pub fn project_expert(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    rows: usize,
    expert: usize,
    inputs: &[&[f32]],
) -> Result<Vec<Vec<f32>>, String> {
    let values = expert_matrix(path, content, name, cols, rows, expert)?;
    inputs
        .iter()
        .map(|input| matvec(&values, cols, input))
        .collect()
}

pub fn expert_matrix(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    rows: usize,
    expert: usize,
) -> Result<Vec<f32>, String> {
    let info = tensor(content, name)?;
    let size = rows
        .checked_mul(cols / 256)
        .and_then(|n| n.checked_mul(210))
        .ok_or("expert size overflow")?;
    let start = expert.checked_mul(size).ok_or("expert offset overflow")?;
    let bytes = read_tensor_range(path, info, start as u64, size as u64, size as u64)?;
    decode_q6_k(&bytes)
}

pub fn project_stream(
    path: &Path,
    content: &Content,
    name: &str,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let info = tensor(content, name)?;
    let row_bytes = (cols / 256).checked_mul(210).ok_or("row size overflow")?;
    let rows = usize::try_from(info.len).map_err(|_| "tensor too large")? / row_bytes;
    let mut output = Vec::with_capacity(rows);
    for first in (0..rows).step_by(256) {
        let count = (rows - first).min(256);
        let start = first
            .checked_mul(row_bytes)
            .ok_or("stream offset overflow")? as u64;
        let size = count.checked_mul(row_bytes).ok_or("stream size overflow")? as u64;
        let bytes = read_tensor_range(path, info, start, size, size)?;
        output.extend(matvec(&decode_q6_k(&bytes)?, cols, input)?);
    }
    Ok(output)
}
