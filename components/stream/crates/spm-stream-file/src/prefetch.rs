//! A bounded background-prefetching parameter stream.

use spm_stream::{StreamError, WeightStream};
use std::{
    io,
    path::{Path, PathBuf},
    sync::mpsc::Receiver,
    thread::JoinHandle,
};

use crate::{
    DEFAULT_CAPACITY,
    worker::{Chunk, spawn_reader},
};

/// A forward-only file stream that overlaps filling the next buffer with work
/// performed by the consumer on the current buffer.
///
/// A rendezvous channel limits residency to two chunks: one held by the
/// consumer and one being filled (or waiting to be handed over) by the worker.
pub struct PrefetchFileWeightStream {
    path: PathBuf,
    capacity: usize,
    receiver: Option<Receiver<Chunk>>,
    worker: Option<JoinHandle<()>>,
    current: Vec<u8>,
    at: usize,
    exhausted: bool,
}

impl PrefetchFileWeightStream {
    /// Opens a stream with the default 64-KiB chunk size.
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StreamError> {
        Self::with_capacity(path, DEFAULT_CAPACITY)
    }

    /// Opens a stream with two chunks of at most `capacity` bytes resident.
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the file cannot be opened.
    pub fn with_capacity(path: impl AsRef<Path>, capacity: usize) -> Result<Self, StreamError> {
        let path = path.as_ref().to_path_buf();
        let capacity = capacity.max(1);
        let (receiver, worker) = spawn_reader(&path, capacity)?;
        Ok(Self {
            path,
            capacity,
            receiver: Some(receiver),
            worker: Some(worker),
            current: Vec::new(),
            at: 0,
            exhausted: false,
        })
    }

    fn stop(&mut self) {
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            worker.join().ok();
        }
    }

    fn refill(&mut self) -> Result<bool, StreamError> {
        let received = self
            .receiver
            .as_ref()
            .ok_or_else(|| io::Error::other("prefetch worker is not running"))?
            .recv()
            .map_err(|_| io::Error::other("prefetch worker stopped"))??;
        self.current = received;
        self.at = 0;
        if self.current.is_empty() {
            self.exhausted = true;
            return Ok(false);
        }
        Ok(true)
    }
}

impl WeightStream for PrefetchFileWeightStream {
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError> {
        if self.at == self.current.len() && (self.exhausted || !self.refill()?) {
            return Ok(0);
        }
        let taken = (self.current.len() - self.at).min(dst.len());
        dst[..taken].copy_from_slice(&self.current[self.at..self.at + taken]);
        self.at += taken;
        Ok(taken)
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        self.stop();
        let (receiver, worker) = spawn_reader(&self.path, self.capacity)?;
        self.receiver = Some(receiver);
        self.worker = Some(worker);
        self.current.clear();
        self.at = 0;
        self.exhausted = false;
        Ok(())
    }
}

impl Drop for PrefetchFileWeightStream {
    fn drop(&mut self) {
        self.stop();
    }
}
