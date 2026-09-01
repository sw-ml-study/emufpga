//! Extraction orchestration and manifest rendering.

use spm_checkpoint_source::{TensorSource, open, read_tensor};
use std::{fmt::Write, fs, path::Path};

/// Encoding of emitted blobs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoding {
    F32,
    Bf16,
}

/// Extraction totals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Summary {
    pub tensors: usize,
    pub parameters: usize,
    pub bytes: usize,
}

/// Extract a supported checkpoint into raw blobs and `manifest.tsv`.
///
/// # Errors
/// Returns an error for invalid checkpoint metadata or any failed file operation.
pub fn extract(source: &Path, output: &Path, encoding: Encoding) -> Result<Summary, String> {
    let mut tensors = open(source)?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let mut manifest = String::from("# name\tshape\tdtype\tblob\telements\n");
    let mut parameters = 0usize;
    for (index, tensor) in tensors.iter().enumerate() {
        parameters = parameters
            .checked_add(write_tensor(
                output,
                &mut manifest,
                tensor,
                index,
                encoding,
            )?)
            .ok_or("checkpoint parameter count overflows")?;
    }
    fs::write(output.join("manifest.tsv"), manifest).map_err(|error| error.to_string())?;
    let width = if encoding == Encoding::Bf16 { 2 } else { 4 };
    summary(tensors.len(), parameters, width)
}

fn summary(tensors: usize, parameters: usize, width: usize) -> Result<Summary, String> {
    let bytes = parameters
        .checked_mul(width)
        .ok_or("checkpoint byte count overflows")?;
    Ok(Summary {
        tensors,
        parameters,
        bytes,
    })
}

fn write_tensor(
    output: &Path,
    manifest: &mut String,
    tensor: &TensorSource,
    index: usize,
    encoding: Encoding,
) -> Result<usize, String> {
    let elements = tensor.shape.iter().product::<usize>();
    let blob = format!("{index:03}.bin");
    let raw = read_tensor(tensor)?;
    let bytes = crate::transform::convert(
        &raw,
        tensor.dtype,
        &tensor.shape,
        encoding == Encoding::Bf16,
    );
    fs::write(output.join(&blob), bytes).map_err(|error| error.to_string())?;
    manifest.push_str(&render_entry(tensor, &blob, elements, encoding));
    Ok(elements)
}

fn render_entry(tensor: &TensorSource, blob: &str, elements: usize, encoding: Encoding) -> String {
    let mut line = String::new();
    let shape = tensor
        .shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let dtype = if encoding == Encoding::Bf16 {
        "bf16"
    } else {
        "f32"
    };
    writeln!(
        line,
        "{}\t{}\t{}\t{}\t{}",
        tensor.name, shape, dtype, blob, elements
    )
    .expect("writing to a string cannot fail");
    line
}
