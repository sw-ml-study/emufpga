//! Dispatch and `PyTorch` state-dict decoding.

use crate::{TensorSource, model::torch_source};
use repugnant_pickle::RepugnantTorchTensors;
use std::path::Path;

/// Inspect a checkpoint without loading its tensor payloads.
///
/// # Errors
/// Returns an error for an unsupported container, dtype, or malformed header.
pub fn open(path: &Path) -> Result<Vec<TensorSource>, String> {
    match path.extension().and_then(|value| value.to_str()) {
        Some("safetensors") => crate::safetensors::open(path),
        Some("pt" | "pth") => open_torch(path),
        _ => Err(format!(
            "{}: expected .pt, .pth, or .safetensors",
            path.display()
        )),
    }
}

fn open_torch(path: &Path) -> Result<Vec<TensorSource>, String> {
    RepugnantTorchTensors::new_from_file(path)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|tensor| torch_source(path, tensor))
        .collect()
}
