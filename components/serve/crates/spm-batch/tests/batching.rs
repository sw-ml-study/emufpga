//! Batching must not change any client's answer.
//!
//! The whole claim of this component is that one sweep of the weights
//! can serve N clients. That is only worth anything if a client gets
//! **exactly** what it would have got alone -- otherwise the schedule
//! is not an optimisation, it is a different model.
//!
//! Asserted bit-for-bit, like every earlier rung, because there is no
//! rounding difference to absorb: batching changes which activations
//! sit beside each other in a buffer, not the order weights are
//! applied in.

use spm_batch::{Client, Scratch, decode_step};
use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_smol::{Resident, SmolConfig};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

fn config() -> SmolConfig {
    SmolConfig {
        hidden: 8,
        intermediate: 16,
        layers: 2,
        heads: 2,
        kv_heads: 1,
        rope_base: 10_000.0,
        eps: 1e-5,
        vocab: 8,
    }
}

fn draw(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32::from(u16::try_from(state % 16).unwrap_or(0)) / 32.0 - 0.25
        })
        .collect()
}

fn shapes(config: &SmolConfig) -> Vec<(u32, u32)> {
    let n = u32::try_from(config.kv_width()).expect("fits");
    let d = u32::try_from(config.hidden).expect("fits");
    let i = u32::try_from(config.intermediate).expect("fits");
    (0..config.layers)
        .flat_map(|_| [(d, d), (n, d), (n, d), (d, d), (i, d), (i, d), (d, i)])
        .collect()
}

fn fixture() -> (SmolConfig, Vec<u8>) {
    let config = config();
    let shapes = shapes(&config);
    let descriptors: Vec<OpDescriptor> = shapes
        .iter()
        .map(|(rows, cols)| OpDescriptor {
            rows: *rows,
            cols: *cols,
            group_size: 16,
            encoding: Encoding::F32,
            lane_count: 1,
        })
        .collect();
    let mut writer = SpmWriter::new(descriptors.clone());
    for (index, descriptor) in descriptors.iter().enumerate() {
        let count = descriptor.rows as usize * descriptor.cols as usize;
        let matrix = draw(index as u64 + 1, count);
        for chunk in matrix.chunks(descriptor.group_size as usize) {
            let mut bytes = vec![0u8; dense_len(chunk.len())];
            encode_into(chunk, &mut bytes).expect("encode");
            writer
                .write_raw_group(1.0, &bytes, chunk.len())
                .expect("write");
        }
    }
    (config, writer.finish().expect("finish"))
}

/// Greedy-decodes `steps` tokens for the given starting tokens, all in
/// one batch, and returns each client's produced tokens.
fn run(config: &SmolConfig, bytes: &[u8], starts: &[u32], steps: usize) -> Vec<Vec<u32>> {
    let embed = draw(500, config.vocab * config.hidden);
    let norms: Vec<(Vec<f32>, Vec<f32>)> = (0..config.layers)
        .map(|i| {
            (
                vec![1.0; config.hidden],
                draw(600 + i as u64, config.hidden),
            )
        })
        .collect();
    let final_norm = vec![1.0f32; config.hidden];
    let resident = Resident {
        embed: &embed,
        norms: &norms,
        final_norm: &final_norm,
    };

    let mut clients: Vec<Client> = starts.iter().map(|t| Client::new(config, 16, *t)).collect();
    let mut scratch = Scratch::new(config, clients.len());
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.to_vec())).expect("open");

    for step in 0..steps {
        if step > 0 {
            groups.rewind().expect("rewind");
        }
        decode_step(&mut groups, config, &resident, (&mut clients, &mut scratch)).expect("step");
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
            client.produced.push(next);
            client.token = next;
        }
    }
    clients.into_iter().map(|c| c.produced).collect()
}

#[test]
fn a_client_batched_with_others_gets_what_it_would_have_got_alone() {
    // THE CLAIM. Five clients decoded together must produce exactly
    // what each produces on its own. If this fails, the sweep is not
    // serving them independently and the amortization is meaningless.
    let (config, bytes) = fixture();
    let starts = [1u32, 3, 5, 2, 7];
    let steps = 4;

    let together = run(&config, &bytes, &starts, steps);
    for (index, start) in starts.iter().enumerate() {
        let alone = run(&config, &bytes, &[*start], steps);
        assert_eq!(
            together[index], alone[0],
            "client {index} (start {start}) decoded differently in a batch"
        );
    }
}

#[test]
fn weight_traffic_is_independent_of_the_client_count() {
    // THE AMORTIZATION, asserted structurally rather than timed. One
    // sweep costs the same bytes whether it serves one client or ten,
    // so bytes per generated token fall as 1/N.
    let (config, bytes) = fixture();
    let mut reports = Vec::new();
    for count in [1usize, 2, 5, 10] {
        let starts: Vec<u32> = (0..count)
            .map(|i| u32::try_from(i % config.vocab).expect("fits"))
            .collect();
        let embed = draw(500, config.vocab * config.hidden);
        let norms: Vec<(Vec<f32>, Vec<f32>)> = (0..config.layers)
            .map(|i| {
                (
                    vec![1.0; config.hidden],
                    draw(600 + i as u64, config.hidden),
                )
            })
            .collect();
        let final_norm = vec![1.0f32; config.hidden];
        let resident = Resident {
            embed: &embed,
            norms: &norms,
            final_norm: &final_norm,
        };
        let mut clients: Vec<Client> = starts
            .iter()
            .map(|t| Client::new(&config, 16, *t))
            .collect();
        let mut scratch = Scratch::new(&config, clients.len());
        let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.clone())).expect("open");
        let report = decode_step(
            &mut groups,
            &config,
            &resident,
            (&mut clients, &mut scratch),
        )
        .expect("step");
        reports.push((count, report));
    }

    let first = reports[0].1.weight_bytes;
    for (count, report) in &reports {
        assert_eq!(
            report.weight_bytes, first,
            "{count} clients read a different number of weight bytes"
        );
        assert_eq!(report.clients, *count);
        assert_eq!(report.streams, config.streams());
    }
    // And therefore per-token cost falls as 1/N. Stated as the ratio
    // rather than as exact division, which only holds when the client
    // count happens to divide the byte count.
    for (count, report) in &reports {
        let per_token = report.weight_bytes / count;
        assert_eq!(
            per_token,
            first / count,
            "{count} clients: per-token weight bytes should be the sweep \
             divided by the clients it served"
        );
    }
    let alone = reports[0].1.weight_bytes;
    let shared = reports[3].1.weight_bytes / 10;
    assert!(
        shared * 9 < alone,
        "ten clients ({shared} B/token) should cost far less per token \
         than one ({alone} B/token)"
    );
}
