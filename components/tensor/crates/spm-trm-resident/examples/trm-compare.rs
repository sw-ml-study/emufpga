//! Streamed against conventional resident, on the real TRM weights.
//!
//!     cargo run --release -p spm-trm-resident --example trm-compare \
//!         -- <trm-ordered.spm> [positions] [repeats]
//!
//! An example rather than a test, for the same reason as `trm-xcheck`:
//! it needs 27 MB of weights, and weights never enter this repository.
//! The result is recorded in docs/results.md.
//!
//! Three configurations, not two. A file-versus-resident number alone
//! conflates the cost of streaming with the cost of storage; the
//! memory-backed stream separates them, because it keeps the streaming
//! discipline and removes the IO.

use spm_stream_file::FileWeightStream;
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_stream_metrics::widen;
use spm_trm::{Layer, TrmConfig};
use spm_trm_resident::{ResidentLayer, ResidentWeights};
use std::time::{Duration, Instant};

/// Deterministic inputs, so every configuration sees the same state.
fn draw(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            f32::from(u16::try_from((state >> 40) & 0xFFFF).unwrap_or(0)) / 32768.0 - 1.0
        })
        .collect()
}

/// Best of `repeats`, which is the honest statistic for a timing whose
/// noise is one-sided: scheduling and cache eviction can only ever
/// make a run slower.
fn best(repeats: usize, mut run: impl FnMut() -> Duration) -> Duration {
    (0..repeats).map(|_| run()).min().unwrap_or_default()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let spm = args
        .next()
        .expect("usage: trm-compare <spm> [positions] [repeats]");
    let positions: usize = args.next().map_or(8, |v| v.parse().expect("positions"));
    let repeats: usize = args.next().map_or(5, |v| v.parse().expect("repeats"));

    let config = TrmConfig::default();
    let input = draw(7, positions * config.hidden);
    let bytes = std::fs::read(&spm).expect("spm");

    // 3. Conventional resident: load once, then address at random.
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.clone())).expect("open");
    let load = Instant::now();
    let weights = ResidentWeights::load(&mut groups).expect("load");
    let load = load.elapsed();
    let mut layers: Vec<_> = (0..config.layers)
        .map(|_| ResidentLayer::new(&config, positions))
        .collect();
    let mut resident_out = input.clone();
    let resident_time = best(repeats, || {
        let mut state = input.clone();
        let at = Instant::now();
        spm_trm_resident::forward(&weights, &config, &mut state, &mut layers).expect("resident");
        let took = at.elapsed();
        resident_out = state;
        took
    });

    // 2. Streamed from memory: streaming discipline, no IO.
    let mut streamed_layers: Vec<_> = (0..config.layers)
        .map(|_| Layer::new(&config, positions))
        .collect();
    let mut mem_out = input.clone();
    let mut group_bytes = 0;
    let mem_time = best(repeats, || {
        let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.clone())).expect("open");
        group_bytes = groups.resident_parameter_bytes();
        let mut state = input.clone();
        let at = Instant::now();
        spm_trm::forward(&mut groups, &config, &mut state, &mut streamed_layers).expect("mem");
        let took = at.elapsed();
        mem_out = state;
        took
    });

    // 1. Streamed from file: parameters never all resident.
    let mut file_out = input.clone();
    let mut read_per_forward = 0u64;
    let file_time = best(repeats, || {
        let stream = FileWeightStream::open(&spm).expect("open file");
        let mut groups = GroupStream::open(stream).expect("open");
        let mut state = input.clone();
        let at = Instant::now();
        let report =
            spm_trm::forward(&mut groups, &config, &mut state, &mut streamed_layers).expect("file");
        let took = at.elapsed();
        read_per_forward = report.weights_read * 4;
        file_out = state;
        took
    });

    let exact = resident_out == mem_out && mem_out == file_out;
    let mismatches = resident_out
        .iter()
        .zip(&file_out)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    println!("positions            {positions}");
    println!("repeats              {repeats}");
    println!("bit-exact all three  {exact}  (mismatched floats: {mismatches})");
    println!();
    println!("                     time/forward   parameter bytes resident");
    println!(
        "resident             {resident_time:>12.3?}   {:>12}",
        weights.parameter_bytes()
    );
    println!("streamed (memory)    {mem_time:>12.3?}   {group_bytes:>12}");
    println!("streamed (file)      {file_time:>12.3?}   {group_bytes:>12}");
    println!();
    println!("resident load once   {load:.3?}");
    println!(
        "residency ratio      {:.6}",
        widen(u64::try_from(group_bytes).unwrap_or(u64::MAX))
            / widen(u64::try_from(weights.parameter_bytes()).unwrap_or(u64::MAX))
    );
    println!();
    // What a real parameter store would have to deliver to keep this
    // engine fed. The rotating region is re-read once per L_level
    // call, so the traffic is 15x the model per forward -- and the
    // demanded RATE falls as batch rises, because the same traffic is
    // spread over more compute. This is the number that decides
    // whether a cheap sequential store can keep up.
    let seconds = file_time.as_secs_f64();
    println!("read per forward     {read_per_forward} bytes");
    println!(
        "store bandwidth      {:.1} MB/s demanded at this batch",
        widen(read_per_forward) / seconds / 1.0e6
    );
}
