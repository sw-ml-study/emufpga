//! Safetensors header decoding and range validation.

use crate::{DType, TensorSource};
use serde_json::Value;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

pub(crate) fn open(path: &Path) -> Result<Vec<TensorSource>, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let file_len = file.metadata().map_err(|error| error.to_string())?.len();
    let (body, header) = read_header(&mut file, file_len)?;
    let values: BTreeMap<String, Value> =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    values
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| tensor(path, body, file_len, name, &value))
        .collect()
}

fn read_header(file: &mut File, file_len: u64) -> Result<(u64, Vec<u8>), String> {
    let mut size = [0; 8];
    file.read_exact(&mut size)
        .map_err(|error| error.to_string())?;
    let header_len = u64::from_le_bytes(size);
    let body = header_len.checked_add(8).ok_or("header length overflows")?;
    if body > file_len {
        return Err("safetensors header exceeds file".into());
    }
    let size = usize::try_from(header_len).map_err(|error| error.to_string())?;
    let mut header = Vec::new();
    header
        .try_reserve_exact(size)
        .map_err(|error| error.to_string())?;
    header.resize(size, 0);
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    Ok((body, header))
}

fn tensor(
    path: &Path,
    body: u64,
    file_len: u64,
    name: String,
    value: &Value,
) -> Result<TensorSource, String> {
    let dtype = DType::from_safetensors(value["dtype"].as_str(), &name)?;
    let shape = serde_json::from_value(value["shape"].clone()).map_err(|e| e.to_string())?;
    let bounds: Vec<u64> =
        serde_json::from_value(value["data_offsets"].clone()).map_err(|error| error.to_string())?;
    let (offset, length) = source_range(&name, body, file_len, &bounds)?;
    Ok(TensorSource {
        name,
        dtype,
        shape,
        path: path.to_path_buf(),
        offset,
        length,
    })
}

fn source_range(
    name: &str,
    body: u64,
    file_len: u64,
    bounds: &[u64],
) -> Result<(u64, usize), String> {
    let [start, end] = bounds else {
        return Err(format!("{name}: bad data offsets"));
    };
    let length = end
        .checked_sub(*start)
        .ok_or_else(|| format!("{name}: reversed offsets"))?;
    let offset = body
        .checked_add(*start)
        .ok_or_else(|| format!("{name}: offset overflows"))?;
    let finish = body
        .checked_add(*end)
        .ok_or_else(|| format!("{name}: end overflows"))?;
    if finish > file_len {
        return Err(format!("{name}: data exceeds checkpoint"));
    }
    let length = usize::try_from(length).map_err(|error| error.to_string())?;
    Ok((offset, length))
}
