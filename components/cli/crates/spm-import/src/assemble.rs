//! Framing extracted blobs into a `.spm` file.

use crate::manifest::{ImportError, Tensor};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_walk::Cursor;
use std::path::Path;

/// Weights per scale group for the f32 encoding.
///
/// The scale is inert for f32 (the weights carry their own
/// magnitude), so group size controls nothing but granularity, and it
/// trades two things against each other. Each group costs four bytes
/// of unused scale, and the reader's reusable buffer is sized to the
/// widest group.
///
/// 1024 puts the scale overhead at 0.1% of payload and the resident
/// buffer at 4 KiB -- small enough that parameter residency stays
/// negligible, large enough that per-group overhead is noise. Nothing
/// about the format requires this number; it is a default, and a
/// caller wanting a different granularity can build descriptors
/// itself.
pub const GROUP_SIZE: u32 = 1024;

/// Builds the stream directory for `tensors`.
#[must_use]
pub fn descriptors(tensors: &[Tensor]) -> Vec<OpDescriptor> {
    tensors
        .iter()
        .map(|t| {
            let (rows, cols) = t.stream_shape();
            OpDescriptor {
                rows,
                cols,
                group_size: GROUP_SIZE,
                encoding: Encoding::F32,
                lane_count: 1,
            }
        })
        .collect()
}

/// Reads each tensor's blob and frames it into a `.spm` file.
///
/// The blobs are already little-endian `f32`, which is exactly the
/// wire format of the f32 encoding, so the bytes pass through
/// untouched. No weight value is read, reordered or rounded here.
///
/// # Errors
/// Returns [`ImportError`] if a blob cannot be read, is not the size
/// its shape implies, or the writer rejects the result.
pub fn assemble(tensors: &[Tensor], dir: &Path) -> Result<Vec<u8>, ImportError> {
    let descriptors = descriptors(tensors);
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    for tensor in tensors {
        let path = dir.join(&tensor.blob);
        let bytes = std::fs::read(&path).map_err(|e| ImportError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        check_size(tensor, bytes.len())?;
        emit(&mut writer, &mut cursor, &descriptors, &bytes)?;
    }
    writer.finish().map_err(|e| ImportError::Format {
        detail: e.to_string(),
    })
}

/// Rejects a blob that does not match its declared shape.
fn check_size(tensor: &Tensor, found: usize) -> Result<(), ImportError> {
    let (rows, cols) = tensor.stream_shape();
    let expected = rows as usize * cols as usize * 4;
    if found == expected {
        return Ok(());
    }
    Err(ImportError::BlobSize {
        name: tensor.name.clone(),
        expected,
        found,
    })
}

/// Writes one tensor's blob as a run of groups.
fn emit(
    writer: &mut SpmWriter,
    cursor: &mut Cursor,
    descriptors: &[OpDescriptor],
    bytes: &[u8],
) -> Result<(), ImportError> {
    let mut at = 0usize;
    while let Some(count) = cursor.group_len(descriptors) {
        let width = count as usize * 4;
        // The scale is inert for f32; 1.0 is written so a reader that
        // multiplies by it unconditionally still gets the weights back.
        writer
            .write_raw_group(1.0, &bytes[at..at + width], count as usize)
            .map_err(|e| ImportError::Format {
                detail: e.to_string(),
            })?;
        at += width;
        cursor.advance(descriptors);
        if at >= bytes.len() {
            return Ok(());
        }
    }
    Ok(())
}
