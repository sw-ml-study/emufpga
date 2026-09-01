//! Reading through the limiter.

use crate::throttle::Throttle;
use spm_stream::{StreamError, WeightStream};

impl<S: WeightStream> WeightStream for Throttle<S> {
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError> {
        let taken = self.inner.next_block(dst)?;
        self.served += taken as u64;
        self.pace();
        Ok(taken)
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        self.inner.rewind()
    }
}
