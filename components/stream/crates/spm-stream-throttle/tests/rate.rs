//! The limiter must actually limit, and must report the wait.

use spm_stream::WeightStream;
use spm_stream_mem::MemoryWeightStream;
use spm_stream_metrics::widen;
use spm_stream_throttle::Throttle;
use std::time::Instant;

/// Drains `stream`, returning the bytes read.
fn drain(stream: &mut impl WeightStream) -> usize {
    let mut buffer = [0u8; 4096];
    let mut total = 0;
    loop {
        let taken = stream.next_block(&mut buffer).expect("read");
        if taken == 0 {
            return total;
        }
        total += taken;
    }
}

#[test]
fn a_throttled_stream_takes_at_least_bytes_over_rate() {
    // 256 KiB at 1 MB/s cannot finish faster than a quarter second,
    // whatever the machine underneath is doing.
    let bytes = vec![7u8; 256 * 1024];
    let expected = bytes.len();
    let mut stream = Throttle::new(MemoryWeightStream::new(bytes), 1.0e6);

    let started = Instant::now();
    let read = drain(&mut stream);
    let elapsed = started.elapsed();

    assert_eq!(read, expected, "throttling must not change what is read");
    let floor = widen(u64::try_from(expected).expect("fits")) / 1.0e6;
    assert!(
        elapsed.as_secs_f64() > floor * 0.9,
        "{expected} B at 1 MB/s took {elapsed:?}, expected at least {floor:.3}s"
    );
    assert!(
        stream.stalled().as_secs_f64() > floor * 0.5,
        "most of that time should be reported as stall, got {:?}",
        stream.stalled()
    );
}

#[test]
fn an_unlimited_throttle_does_not_wait() {
    // Rate zero means no limit, so a sweep can include "as fast as the
    // machine allows" without a special case at the call site.
    let bytes = vec![7u8; 256 * 1024];
    let expected = bytes.len();
    let mut stream = Throttle::new(MemoryWeightStream::new(bytes), 0.0);
    assert_eq!(drain(&mut stream), expected);
    assert_eq!(stream.stalled(), std::time::Duration::ZERO);
}

#[test]
fn throttling_changes_timing_and_nothing_else() {
    // The bytes delivered must be identical, or the measurement is
    // measuring a different stream.
    let bytes: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
    let mut fast = Throttle::new(MemoryWeightStream::new(bytes.clone()), 0.0);
    let mut slow = Throttle::new(MemoryWeightStream::new(bytes), 4.0e6);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut buf = [0u8; 512];
    loop {
        let n = fast.next_block(&mut buf).expect("fast");
        if n == 0 {
            break;
        }
        a.extend_from_slice(&buf[..n]);
    }
    loop {
        let n = slow.next_block(&mut buf).expect("slow");
        if n == 0 {
            break;
        }
        b.extend_from_slice(&buf[..n]);
    }
    assert_eq!(a, b, "a throttle must not alter the byte stream");
}
