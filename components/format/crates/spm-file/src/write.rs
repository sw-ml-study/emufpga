//! Building a `.spm` file, one scale group at a time.

use crate::error::FileError;
use spm_codec::{Ternary, encode_into, packed_len};
use spm_header::{Header, render as render_header};
use spm_layout::{OpDescriptor, render as render_descriptor};
use spm_walk::Cursor;

/// Writes a `.spm` file group by group, in stream order.
///
/// The header and directory are emitted up front, so every descriptor
/// must be known before any weight is written. That is the constraint
/// the hardware has too: the engine is configured for an operation
/// before its stream starts.
pub struct SpmWriter {
    descriptors: Vec<OpDescriptor>,
    out: Vec<u8>,
    cursor: Cursor,
}

impl SpmWriter {
    /// Starts a file declaring `descriptors`, emitting the header and
    /// stream directory immediately.
    #[must_use]
    pub fn new(descriptors: Vec<OpDescriptor>) -> Self {
        let count = u32::try_from(descriptors.len()).unwrap_or(u32::MAX);
        let mut out = Vec::from(render_header(&Header::new(count)));
        for descriptor in &descriptors {
            out.extend_from_slice(&render_descriptor(descriptor));
        }
        let cursor = Cursor::new(&descriptors);
        Self {
            descriptors,
            out,
            cursor,
        }
    }

    /// Appends one scale group: the scale, then its packed weights.
    ///
    /// # Errors
    /// Returns [`FileError::GroupLen`] if `weights` does not match the
    /// length the layout fixes for this group, or
    /// [`FileError::Incomplete`] if every declared stream is already
    /// written.
    pub fn write_group(&mut self, scale: f32, weights: &[Ternary]) -> Result<(), FileError> {
        let expected = self
            .cursor
            .group_len(&self.descriptors)
            .ok_or(FileError::Incomplete {
                written: self.cursor.stream,
                declared: self.descriptors.len(),
            })?;
        if weights.len() != expected as usize {
            return Err(FileError::GroupLen {
                expected,
                offered: weights.len(),
            });
        }
        self.out.extend_from_slice(&scale.to_le_bytes());
        let at = self.out.len();
        self.out.resize(at + packed_len(weights.len()), 0);
        encode_into(weights, &mut self.out[at..])?;
        self.cursor.advance(&self.descriptors);
        Ok(())
    }

    /// Finishes the file, or reports which streams are still missing.
    ///
    /// # Errors
    /// Returns [`FileError::Incomplete`] if any declared stream has
    /// unwritten groups. A short `.spm` file is more dangerous than a
    /// missing one: the engine would read plausible garbage off the
    /// end rather than fail.
    pub fn finish(self) -> Result<Vec<u8>, FileError> {
        if self.cursor.stream == self.descriptors.len() {
            Ok(self.out)
        } else {
            Err(FileError::Incomplete {
                written: self.cursor.stream,
                declared: self.descriptors.len(),
            })
        }
    }
}
