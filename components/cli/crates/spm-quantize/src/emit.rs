//! Writing a quantized matrix out as `.spm` bytes.

use crate::quantize::Quantized;
use spm_file::{FileError, SpmWriter};
use spm_walk::Cursor;

/// Serializes `quantized` to `.spm` bytes.
///
/// Goes through the real [`SpmWriter`] rather than assembling bytes
/// here, so the CLI cannot drift from the format the library defines.
///
/// # Errors
/// Returns [`FileError`] if the group lengths disagree with the
/// descriptor, which would mean [`crate::quantize`] and the layout
/// crate have diverged.
pub fn write_spm(quantized: &Quantized) -> Result<Vec<u8>, FileError> {
    let descriptors = vec![quantized.descriptor];
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    let (mut at, mut group) = (0usize, 0usize);
    while let Some(len) = cursor.group_len(&descriptors) {
        let len = len as usize;
        writer.write_group(quantized.scales[group], &quantized.weights[at..at + len])?;
        cursor.advance(&descriptors);
        at += len;
        group += 1;
    }
    writer.finish()
}
