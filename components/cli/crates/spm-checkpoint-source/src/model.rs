//! Source tensor descriptions.

use repugnant_pickle::{RepugnantTorchTensor, TensorType};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Numeric formats accepted at the framework boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F32,
    Bf16,
    F16,
}

impl DType {
    /// Bytes occupied by one source element.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }

    pub(crate) fn from_safetensors(value: Option<&str>, name: &str) -> Result<Self, String> {
        match value {
            Some("F32") => Ok(Self::F32),
            Some("BF16") => Ok(Self::Bf16),
            Some("F16") => Ok(Self::F16),
            other => Err(format!("{name}: unsupported safetensors dtype {other:?}")),
        }
    }
}

/// One tensor's location inside a checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSource {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub path: PathBuf,
    pub offset: u64,
    pub length: usize,
}

/// Read one tensor's exact source bytes.
///
/// # Errors
/// Returns an error when the checkpoint cannot be opened, sought, or read.
pub fn read_tensor(source: &TensorSource) -> Result<Vec<u8>, String> {
    let mut file = File::open(&source.path).map_err(|error| error.to_string())?;
    let file_len = file.metadata().map_err(|error| error.to_string())?.len();
    let end = source
        .offset
        .checked_add(source.length as u64)
        .ok_or_else(|| format!("{}: tensor range overflows", source.name))?;
    if end > file_len {
        return Err(format!("{}: tensor range exceeds checkpoint", source.name));
    }
    file.seek(SeekFrom::Start(source.offset))
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(source.length)
        .map_err(|error| error.to_string())?;
    bytes.resize(source.length, 0);
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

pub(crate) fn torch_source(
    path: &Path,
    tensor: RepugnantTorchTensor,
) -> Result<TensorSource, String> {
    let dtype = match tensor.tensor_type {
        TensorType::Float32 => DType::F32,
        TensorType::BFloat16 => DType::Bf16,
        TensorType::Float16 => DType::F16,
        other => {
            return Err(format!(
                "{}: unsupported torch dtype {other:?}",
                tensor.name
            ));
        }
    };
    let elements = tensor
        .shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| format!("{}: shape overflows", tensor.name))?;
    let length = elements
        .checked_mul(dtype.width())
        .ok_or_else(|| format!("{}: byte length overflows", tensor.name))?;
    Ok(TensorSource {
        name: tensor.name,
        dtype,
        shape: tensor.shape,
        path: path.to_path_buf(),
        offset: tensor.absolute_offset,
        length,
    })
}
