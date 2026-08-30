//! A parameter stream reading sequentially through a file.

use crate::buffer::BufferPair;
use spm_stream::{StreamError, WeightStream};
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::Path;

/// Bytes pulled from the file per refill.
///
/// 64 KiB is large enough that syscall overhead is not the story and
/// small enough that parameter residency (`Rp`) stays near zero, which
/// is the property the architecture is trying to demonstrate.
pub const DEFAULT_CAPACITY: usize = 64 * 1024;

/// A parameter stream over a file, read strictly forward.
///
/// The whole file streams, header and directory included, so even
/// metadata arrives sequentially rather than by seeking to an offset.
pub struct FileWeightStream {
    file: File,
    buffers: BufferPair,
    capacity: usize,
}

impl FileWeightStream {
    /// Opens `path` with [`DEFAULT_CAPACITY`] buffers.
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StreamError> {
        Self::with_capacity(path, DEFAULT_CAPACITY)
    }

    /// Opens `path` with buffers of `capacity` bytes each.
    ///
    /// Small capacities are useful in tests, where they force many
    /// buffer boundaries and so exercise the short-read path.
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the file cannot be opened.
    pub fn with_capacity(path: impl AsRef<Path>, capacity: usize) -> Result<Self, StreamError> {
        let capacity = capacity.max(1);
        Ok(Self {
            file: File::open(path)?,
            buffers: BufferPair::new(capacity),
            capacity,
        })
    }
}

impl WeightStream for FileWeightStream {
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError> {
        if self.buffers.is_drained() && self.buffers.refill_from(&mut self.file)? == 0 {
            return Ok(0);
        }
        Ok(self.buffers.drain_into(dst))
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        self.file.seek(SeekFrom::Start(0))?;
        self.buffers = BufferPair::new(self.capacity);
        Ok(())
    }
}
