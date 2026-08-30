//! The file stream must be indistinguishable from the memory stream.
//!
//! If these two ever diverge, every measurement taken against one
//! backend stops predicting the other, and the whole point of having
//! a reference implementation is lost.

use spm_stream::WeightStream;
use spm_stream_file::FileWeightStream;
use spm_stream_mem::MemoryWeightStream;
use std::io::Write;

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

/// Writes `bytes` to a uniquely named scratch file and returns it.
fn scratch(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("emufpga-{name}-{}.spm", std::process::id()));
    let mut file = std::fs::File::create(&path).expect("create");
    file.write_all(bytes).expect("write");
    path
}

#[test]
fn matches_the_memory_stream_byte_for_byte() {
    // Capacities deliberately smaller than the data, and coprime with
    // it, so refills land mid-block and exercise the short-read path.
    let bytes: Vec<u8> = (0..1000u32).map(|v| (v % 251) as u8).collect();
    let path = scratch("equiv", &bytes);
    for capacity in [1, 7, 64, 999, 1000, 4096] {
        for block in [1, 5, 64, 4096] {
            let mut file = FileWeightStream::with_capacity(&path, capacity).expect("open");
            let mut memory = MemoryWeightStream::new(bytes.clone());
            assert_eq!(
                drain(&mut file, block),
                drain(&mut memory, block),
                "capacity {capacity}, block {block}"
            );
        }
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn rewind_preserves_a_custom_buffer_capacity() {
    // Regression: an early draft rebuilt the buffer pair at the
    // default capacity on rewind, silently changing the IO pattern
    // between the first scan and every scan after it.
    let bytes: Vec<u8> = (0..300u32).map(|v| (v % 251) as u8).collect();
    let path = scratch("rewind", &bytes);
    let mut stream = FileWeightStream::with_capacity(&path, 8).expect("open");
    let first = drain(&mut stream, 3);
    stream.rewind().expect("rewind");
    let second = drain(&mut stream, 3);
    assert_eq!(first, bytes);
    assert_eq!(second, bytes);
    std::fs::remove_file(&path).ok();
}

#[test]
fn an_empty_file_is_immediately_exhausted() {
    let path = scratch("empty", &[]);
    let mut stream = FileWeightStream::open(&path).expect("open");
    let mut buffer = [0u8; 16];
    assert_eq!(stream.next_block(&mut buffer).expect("next_block"), 0);
    std::fs::remove_file(&path).ok();
}
