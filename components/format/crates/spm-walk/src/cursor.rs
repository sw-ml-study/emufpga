//! A forward-only position in a file's sequence of scale groups.

use spm_layout::{OpDescriptor, group_count, group_len, weight_count};

/// A position in the (stream, group) sequence.
///
/// Streams declaring zero weights are skipped automatically, so a
/// cursor never rests on a group that does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    /// Index of the current stream.
    pub stream: usize,
    /// Index of the current group within that stream.
    pub group: u64,
}

impl Cursor {
    /// A cursor at the first group that actually exists.
    #[must_use]
    pub fn new(descriptors: &[OpDescriptor]) -> Self {
        let mut cursor = Self {
            stream: 0,
            group: 0,
        };
        cursor.settle(descriptors);
        cursor
    }

    /// Weights in the current group, or `None` past the last stream.
    #[must_use]
    pub fn group_len(&self, descriptors: &[OpDescriptor]) -> Option<u32> {
        let descriptor = descriptors.get(self.stream)?;
        let total = weight_count(descriptor.rows, descriptor.cols);
        Some(group_len(total, descriptor.group_size, self.group))
    }

    /// Moves to the next group, crossing into the next stream at the
    /// end of the current one.
    pub fn advance(&mut self, descriptors: &[OpDescriptor]) {
        if self.stream >= descriptors.len() {
            return;
        }
        self.group += 1;
        self.settle(descriptors);
    }

    /// Rolls forward past any stream with no groups left to read.
    fn settle(&mut self, descriptors: &[OpDescriptor]) {
        while let Some(descriptor) = descriptors.get(self.stream) {
            let total = weight_count(descriptor.rows, descriptor.cols);
            if self.group < group_count(total, descriptor.group_size) {
                return;
            }
            self.stream += 1;
            self.group = 0;
        }
    }
}
