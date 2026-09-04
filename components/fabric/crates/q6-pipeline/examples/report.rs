use q6_pipeline::{BLOCK_BYTES, Config, run};

fn main() {
    let bytes: Vec<_> = (0..BLOCK_BYTES)
        .map(|i| u8::try_from(i % 256).unwrap_or_default())
        .collect();
    let config = Config {
        fifo_bytes: 420,
        fetch_bytes_per_cycle: 16,
        decoder_lanes: 8,
        mac_lanes: 16,
    };
    for selected in [false, true] {
        let outcome = run(&bytes, selected, None, config).expect("known-answer fixture");
        println!(
            "selected={selected} bytes={} total={} fetch={} decode={} mac={} starved={} backpressured={} peak_fifo={}",
            outcome.bytes,
            outcome.cycles.total,
            outcome.cycles.fetch,
            outcome.cycles.decode,
            outcome.cycles.mac,
            outcome.cycles.starved,
            outcome.cycles.backpressured,
            outcome.cycles.peak_fifo_bytes
        );
    }
}
