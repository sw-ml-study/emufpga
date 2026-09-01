//! Serve N clients from one sweep of the weights, on the real model.
//!
//!     cargo run --release -p spm-batch --example serve-demo \
//!         -- <dir> <spm> <extracted> [steps]
//!
//! `<dir>` holds `greedy.json` from `tools/smol-reference/greedy.py`.
//!
//! Two things are checked and one is measured:
//!
//! 1. Each client's greedy tokens match `transformers` decoding the
//!    same prompt alone. The formulas were verified in step 8; what is
//!    new here is the KV cache and the position handling, and an
//!    off-by-one in either produces fluent-looking wrong output.
//! 2. Clients batched together produce what they produce alone.
//! 3. Weight bytes per generated token, against the client count.

use spm_batch::{Client, Scratch, decode_step};
use spm_smol::{Resident, SmolConfig};
use spm_stream_file::FileWeightStream;
use spm_stream_groups::GroupStream;
use spm_stream_metrics::widen;
use spm_stream_throttle::Throttle;

mod support;
use support::{Fixture, load};

/// Runs `prompts` to completion, all in one batch, greedily.
fn serve(
    spm: &str,
    config: &SmolConfig,
    fixture: &Fixture,
    prompts: &[Vec<u32>],
    steps: usize,
    rate: f64,
) -> (
    Vec<Vec<u32>>,
    usize,
    (std::time::Duration, std::time::Duration),
) {
    let resident = Resident {
        embed: &fixture.embed,
        norms: &fixture.norms,
        final_norm: &fixture.final_norm,
    };
    let context = prompts.iter().map(Vec::len).max().unwrap_or(0) + steps + 1;
    let mut clients: Vec<Client> = prompts
        .iter()
        .map(|p| Client::new(config, context, p[0]))
        .collect();
    let mut scratch = Scratch::new(config, clients.len());
    // A throttled store: `rate` bytes per second, 0 meaning unlimited.
    // Every earlier measurement read a page-cached file, so the
    // demanded-bandwidth figures were requirements rather than
    // observations. This makes the store real.
    let throttle = Throttle::new(FileWeightStream::open(spm).expect("open"), rate);
    let stalls = throttle.meter();
    let mut groups = GroupStream::open(throttle).expect("spm");

    let longest = prompts.iter().map(Vec::len).max().unwrap_or(1);
    let total = longest + steps - 1;
    let (mut bytes, mut elapsed) = (0usize, std::time::Duration::ZERO);
    for tick in 0..total {
        if tick > 0 {
            groups.rewind().expect("rewind");
        }
        let started = std::time::Instant::now();
        let report = decode_step(&mut groups, config, &resident, (&mut clients, &mut scratch))
            .expect("step");
        elapsed += started.elapsed();
        bytes = report.weight_bytes;
        let vocab = config.vocab;
        for (index, client) in clients.iter_mut().enumerate() {
            let row = &scratch.logits[index * vocab..(index + 1) * vocab];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
                .expect("nonempty")
                .0;
            let next = u32::try_from(best).expect("fits");
            // Still inside this client's prompt: feed the next prompt
            // token and discard the prediction. Past it: generate.
            let prompt = &prompts[index];
            if tick + 1 < prompt.len() {
                client.token = prompt[tick + 1];
            } else {
                client.produced.push(next);
                client.token = next;
            }
        }
    }
    let produced = clients.iter().map(|c| c.produced.clone()).collect();
    let stalled =
        std::time::Duration::from_nanos(stalls.load(std::sync::atomic::Ordering::Relaxed));
    (produced, bytes, (elapsed, stalled))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: serve-demo <dir> <spm> <extracted> [steps]");
    let spm = args.next().expect("spm");
    let extracted = args.next().expect("extracted");
    let steps: usize = args.next().map_or(8, |v| v.parse().expect("steps"));
    // Store speed in MB/s. 0 means unlimited -- the page-cached
    // baseline every earlier result used.
    let store: f64 = args.next().map_or(0.0, |v| v.parse().expect("store MB/s"));

    let config = SmolConfig::default();
    let fixture = load(&extracted, &config);
    let reference = support::greedy(&dir);

    let prompts: Vec<Vec<u32>> = reference.prompts.clone();
    let (produced, bytes, (elapsed, stalled)) =
        serve(&spm, &config, &fixture, &prompts, steps, store * 1.0e6);

    println!("=== correctness ===");
    for (index, got) in produced.iter().enumerate() {
        let want = &reference.produced[index][..steps.min(reference.produced[index].len())];
        let cut = &got[..want.len().min(got.len())];
        let ok = cut == want;
        println!(
            "client {index}  {}  {cut:?}",
            if ok { "MATCH " } else { "DIFFER" }
        );
        if !ok {
            println!("          want {want:?}");
        }
    }

    println!();
    println!("=== amortization ===");
    println!("weight bytes per sweep   {bytes}");
    println!("clients served           {}", prompts.len());
    println!(
        "bytes per generated token {:.1} MB",
        widen(u64::try_from(bytes).unwrap_or(u64::MAX))
            / widen(u64::try_from(prompts.len()).unwrap_or(1))
            / 1.0e6
    );
    let probe = Client::new(
        &config,
        prompts.iter().map(Vec::len).max().unwrap_or(0) + steps + 1,
        0,
    );
    let per_client: usize = probe.bytes();
    println!(
        "KV cache per client      {:.1} MB",
        widen(u64::try_from(per_client).unwrap_or(u64::MAX)) / 1.0e6
    );
    println!("total decode time        {elapsed:.3?}");
    if store > 0.0 {
        println!("store speed              {store} MB/s");
        println!("stalled on store         {stalled:.3?}");
        println!(
            "store share of wall      {:.1}%",
            100.0 * stalled.as_secs_f64() / elapsed.as_secs_f64()
        );
    } else {
        println!("store speed              unlimited (page cache)");
    }
}
