//! A two-slot buffer pair, drained from one slot while the other
//! stands ready to be filled.

use std::io::{self, Read};

/// Two equally sized slots, one active and being drained.
pub(crate) struct BufferPair {
    slots: [Vec<u8>; 2],
    active: usize,
    filled: usize,
    at: usize,
}

impl BufferPair {
    /// A pair of empty slots, each able to hold `capacity` bytes.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            slots: [vec![0; capacity], vec![0; capacity]],
            active: 0,
            filled: 0,
            at: 0,
        }
    }

    /// Whether the active slot has been fully consumed.
    pub(crate) const fn is_drained(&self) -> bool {
        self.at >= self.filled
    }

    /// Copies what the active slot still holds into `dst`.
    ///
    /// Returns the number of bytes moved, which may be short of
    /// `dst.len()` -- a short read is a buffer boundary, not the end
    /// of the stream.
    pub(crate) fn drain_into(&mut self, dst: &mut [u8]) -> usize {
        let taken = (self.filled - self.at).min(dst.len());
        dst[..taken].copy_from_slice(&self.slots[self.active][self.at..self.at + taken]);
        self.at += taken;
        taken
    }

    /// Swaps slots and fills the newly active one from `source`.
    ///
    /// Returns the bytes read; zero means the source is exhausted.
    /// The swap is what a prefetching backend would overlap: it would
    /// hand back an already-full slot instead of filling one here.
    pub(crate) fn refill_from(&mut self, source: &mut impl Read) -> io::Result<usize> {
        self.active ^= 1;
        self.at = 0;
        self.filled = 0;
        let slot = &mut self.slots[self.active];
        let mut total = 0;
        while total < slot.len() {
            let taken = source.read(&mut slot[total..])?;
            if taken == 0 {
                break;
            }
            total += taken;
        }
        self.filled = total;
        Ok(total)
    }
}
