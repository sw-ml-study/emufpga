use q6_pipeline::{BLOCK_BYTES, Config, Error, checksum, run};

fn config() -> Config {
    Config {
        fifo_bytes: 420,
        fetch_bytes_per_cycle: 16,
        decoder_lanes: 8,
        mac_lanes: 16,
    }
}
fn fixture(blocks: usize) -> Vec<u8> {
    (0..blocks * BLOCK_BYTES)
        .map(|i| {
            u8::try_from(i % 256)
                .unwrap_or_default()
                .wrapping_mul(73)
                .wrapping_add(19)
        })
        .collect()
}

#[test]
fn independent_decoder_matches_the_gguf_oracle() {
    let bytes = fixture(3);
    let got = run(&bytes, true, None, config()).expect("pipeline");
    let expected = spm_gguf::decode_q6_k(&bytes).expect("oracle");
    assert_eq!(got.values, expected);
    assert_eq!(got.bytes, bytes.len());
    assert_eq!(got.events.len(), 3);
    assert!(
        got.events
            .windows(2)
            .all(|pair| pair[0].issued < pair[1].issued)
    );
}

#[test]
fn unselected_experts_cross_pins_without_mac_work() {
    let bytes = fixture(2);
    let got = run(&bytes, false, None, config()).expect("pipeline");
    assert!(got.values.is_empty());
    assert_eq!(got.cycles.mac, 0);
    assert_eq!(got.bytes, bytes.len());
}

#[test]
fn truncation_and_corruption_fail_loudly_when_framed() {
    assert_eq!(
        run(&fixture(1)[..209], true, None, config()).unwrap_err(),
        Error::Truncated
    );
    let mut bytes = fixture(1);
    let expected = checksum(&bytes);
    bytes[17] ^= 1;
    assert_eq!(
        run(&bytes, true, Some(expected), config()).unwrap_err(),
        Error::ChecksumMismatch
    );
}

#[test]
fn slow_fetch_starves_and_fast_fetch_backpressures() {
    let bytes = fixture(4);
    let slow = run(
        &bytes,
        true,
        None,
        Config {
            fetch_bytes_per_cycle: 1,
            ..config()
        },
    )
    .expect("slow");
    let fast = run(
        &bytes,
        true,
        None,
        Config {
            fetch_bytes_per_cycle: 210,
            ..config()
        },
    )
    .expect("fast");
    assert!(slow.cycles.starved > fast.cycles.starved);
    assert!(fast.cycles.backpressured > 0);
    assert_eq!(slow.values, fast.values);
}
