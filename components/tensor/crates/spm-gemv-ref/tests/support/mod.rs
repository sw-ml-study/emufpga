//! Shared helpers: turn a golden case into a `.spm` file.

use spm_codec::Ternary;
use spm_file::SpmWriter;
use spm_vectors::GoldenCase;
use spm_walk::Cursor;

/// Serializes a case's weights and scales into `.spm` bytes.
///
/// Goes through the real writer rather than hand-assembling bytes, so
/// these tests exercise the same path the packer will.
#[must_use]
pub fn to_spm(case: &GoldenCase) -> Vec<u8> {
    let descriptors = vec![case.descriptor];
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    let mut at = 0usize;
    let mut group = 0usize;
    while let Some(len) = cursor.group_len(&descriptors) {
        let len = len as usize;
        let weights: Vec<Ternary> = case.weights[at..at + len].to_vec();
        writer
            .write_group(case.scales[group], &weights)
            .expect("write group");
        cursor.advance(&descriptors);
        at += len;
        group += 1;
    }
    writer.finish().expect("finish")
}
