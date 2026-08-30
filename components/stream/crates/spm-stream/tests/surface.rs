//! A tripwire on the trait's required surface.
//!
//! This is the *enforced* half of the no-seek guarantee. The
//! `compile_fail` doctest on `WeightStream` shows that a seek call
//! does not compile, but `compile_fail` accepts any compile error and
//! so cannot prove *why*.
//!
//! `Minimal` below implements the trait with exactly two methods and
//! nothing else. If a third required method is ever added -- `seek`,
//! `position`, `len`, anything that leaks a location -- this file
//! stops compiling and the change cannot land quietly.
//!
//! A method with a default body could still slip past. That is a much
//! smaller hole than it sounds: a defaulted `seek` would have to be
//! implemented in terms of `next_block` alone, which for a
//! forward-only store is not possible without buffering the entire
//! stream -- exactly the thing the architecture refuses to do.

use spm_stream::{StreamError, WeightStream};

/// The smallest possible implementation: a stream that is always
/// exhausted. Its value is structural, not behavioural.
struct Minimal;

impl WeightStream for Minimal {
    fn next_block(&mut self, _dst: &mut [u8]) -> Result<usize, StreamError> {
        Ok(0)
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        Ok(())
    }
}

#[test]
fn the_required_surface_is_exactly_next_block_and_rewind() {
    // Compiling at all is the assertion. The body just exercises it.
    let mut stream = Minimal;
    let mut buffer = [0u8; 4];
    assert_eq!(stream.next_block(&mut buffer).expect("next_block"), 0);
    assert!(stream.rewind().is_ok());
}

#[test]
fn a_generic_consumer_can_only_move_forward() {
    // An engine written against the trait has this entire vocabulary.
    // There is nothing else to call.
    fn engine(stream: &mut impl WeightStream) -> usize {
        let mut buffer = [0u8; 8];
        let mut total = 0;
        while let Ok(taken) = stream.next_block(&mut buffer) {
            if taken == 0 {
                break;
            }
            total += taken;
        }
        total
    }
    assert_eq!(engine(&mut Minimal), 0);
}
