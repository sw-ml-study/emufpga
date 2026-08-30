//! A parameter stream over an owned byte buffer.

use spm_stream::{StreamError, WeightStream};

/// A parameter stream reading from memory.
///
/// The buffer is private and there is no accessor for it: handing out
/// the slice would hand out random access, defeating the point of the
/// trait.
pub struct MemoryWeightStream {
    bytes: Vec<u8>,
    at: usize,
}

impl MemoryWeightStream {
    /// A stream over `bytes`, positioned at the start.
    #[must_use]
    pub const fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, at: 0 }
    }
}

impl WeightStream for MemoryWeightStream {
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError> {
        let remaining = self.bytes.len() - self.at;
        let taken = remaining.min(dst.len());
        dst[..taken].copy_from_slice(&self.bytes[self.at..self.at + taken]);
        self.at += taken;
        Ok(taken)
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        self.at = 0;
        Ok(())
    }
}
