//! The memory stream is the reference every other backend must match.

use spm_stream::{StreamError, WeightStream};
use spm_stream_mem::MemoryWeightStream;

fn drain(stream: &mut impl WeightStream, block: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buffer = vec![0u8; block];
    loop {
        let taken = stream.next_block(&mut buffer).expect("next_block");
        if taken == 0 {
            return out;
        }
        out.extend_from_slice(&buffer[..taken]);
    }
}

#[test]
fn delivers_every_byte_in_order_whatever_the_block_size() {
    let bytes: Vec<u8> = (0..=255).collect();
    for block in [1, 3, 16, 255, 256, 1024] {
        let mut stream = MemoryWeightStream::new(bytes.clone());
        assert_eq!(drain(&mut stream, block), bytes, "block {block}");
    }
}

#[test]
fn zero_means_exhausted_and_stays_exhausted() {
    let mut stream = MemoryWeightStream::new(vec![1, 2, 3]);
    let mut buffer = [0u8; 8];
    assert_eq!(stream.next_block(&mut buffer).expect("first"), 3);
    assert_eq!(stream.next_block(&mut buffer).expect("second"), 0);
    assert_eq!(stream.next_block(&mut buffer).expect("third"), 0);
}

#[test]
fn rewind_returns_to_the_start_of_the_whole_stream() {
    // rewind is the only backward move the trait allows, and only
    // between operations. The research's rotating parameter store
    // rewinds exactly this way, once per full scan.
    let bytes = vec![9u8; 40];
    let mut stream = MemoryWeightStream::new(bytes.clone());
    assert_eq!(drain(&mut stream, 7), bytes);
    stream.rewind().expect("rewind");
    assert_eq!(drain(&mut stream, 7), bytes);
}

#[test]
fn read_exact_fails_rather_than_returning_short() {
    let mut stream = MemoryWeightStream::new(vec![1, 2, 3]);
    let mut buffer = [0u8; 5];
    let error = stream.read_exact(&mut buffer).expect_err("must fail");
    assert!(
        matches!(
            error,
            StreamError::Truncated {
                needed: 5,
                available: 3
            }
        ),
        "got {error:?}"
    );
}
