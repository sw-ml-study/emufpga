//! The rung's defining property: a forward with no rewind at all.
//!
//! Hermetic, at a reduced width. What is under test is the streaming
//! shape -- 7 streams per layer, read once, start to finish -- which
//! is a property of the architecture rather than of trained values.
//! The reference comparison is the `smol-xcheck` example and its
//! result is in docs/results.md.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_smol::{Layer, Resident, SmolConfig, forward};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

fn config() -> SmolConfig {
    SmolConfig {
        hidden: 8,
        intermediate: 16,
        layers: 2,
        heads: 2,
        kv_heads: 1,
        rope_base: 100_000.0,
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

/// Seven streams per layer, in consumption order: q, k, v, o, gate, up, down.
fn shapes(config: &SmolConfig) -> Vec<(u32, u32)> {
    let d = u32::try_from(config.hidden).expect("fits");
    let i = u32::try_from(config.intermediate).expect("fits");
    let kv = u32::try_from(config.kv_width()).expect("fits");
    (0..config.layers)
        .flat_map(|_| [(d, d), (kv, d), (kv, d), (d, d), (i, d), (i, d), (d, i)])
        .collect()
}

fn to_spm(shapes: &[(u32, u32)], matrices: &[Vec<f32>]) -> Vec<u8> {
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
    for (matrix, descriptor) in matrices.iter().zip(&descriptors) {
        for chunk in matrix.chunks(descriptor.group_size as usize) {
            let mut bytes = vec![0u8; dense_len(chunk.len())];
            encode_into(chunk, &mut bytes).expect("encode");
            writer
                .write_raw_group(1.0, &bytes, chunk.len())
                .expect("write");
        }
    }
    writer.finish().expect("finish")
}

#[test]
fn a_forward_reads_every_stream_once_and_never_rewinds() {
    // THE PROPERTY OF THIS RUNG. TRM rewinds 14 times per forward, HRM
    // 3, BDH 5. SmolLM rewinds ZERO times: 30 distinct layers, each
    // weight read once, and the pass ends at the end of the stream.
    let config = config();
    let shapes = shapes(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 1, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, &matrices);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    assert_eq!(groups.descriptors.len(), config.streams(), "7 per layer");

    let positions = 3;
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

    let vocab = u32::try_from(config.vocab).expect("fits");
    let tokens: Vec<u32> = (0..u32::try_from(positions).expect("fits"))
        .map(|t| t % vocab)
        .collect();
    let mut state = vec![0.0f32; positions * config.hidden];
    let mut logits = vec![0.0f32; positions * config.vocab];
    let io = (&tokens[..], &mut state[..], &mut logits[..]);
    forward(
        &mut groups,
        &config,
        &resident,
        &mut Layer::new(&config, positions),
        io,
    )
    .expect("forward");

    assert!(state.iter().all(|v| v.is_finite()), "state went non-finite");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits went non-finite"
    );
    // The stream is exhausted: everything was consumed, nothing left.
    assert!(
        groups.next_group().is_none(),
        "a forward should end at the end of the stream, with no rewind \
         and nothing unread"
    );
}

#[test]
fn tied_embeddings_make_a_fifth_of_this_model_unstreamable() {
    // The rung's cost, at the REAL configuration. An embedding is
    // gathered by token id, so it cannot be swept to serve one token,
    // and SmolLM ties it to the output projection. Compare 0.13%
    // resident for TRM and 0.05% for HRM: a research model with a tiny
    // vocabulary flatters this architecture, a real one does not.
    let config = SmolConfig::default();
    let embed = config.vocab * config.hidden;
    let per_layer = 2 * config.hidden * config.hidden
        + 2 * config.kv_width() * config.hidden
        + 3 * config.intermediate * config.hidden;
    let streamed = config.layers * per_layer;
    let norms = config.layers * 2 * config.hidden + config.hidden;
    let total = streamed + embed + norms;

    assert_eq!(total, 134_515_008, "published parameter count");
    assert_eq!(streamed, 106_168_320, "seven matrices x 30 layers");
    let resident_percent = (embed + norms) * 100 / total;
    assert_eq!(resident_percent, 21, "a fifth of the model stays in RAM");
}
