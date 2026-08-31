//! BDH's rotating store, and the activation cost it comes with.
//!
//! Hermetic: a narrow BDH at synthetic weights, not the 101 MB export.
//! The structure under test -- one parameter set applied `n_layer`
//! times, `lm_head` reached by reading on -- is a property of the
//! architecture, not of trained values. The reference comparison lives
//! in the `bdh-xcheck` example and its result is in docs/results.md.

use spm_bdh::{BdhConfig, Level, forward};
use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

/// A narrow BDH: same structure, small enough to run in a test.
fn config() -> BdhConfig {
    BdhConfig {
        n_layer: 3,
        hidden: 8,
        heads: 2,
        multiplier: 4,
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

/// The nine rotating streams, then `lm_head`, in consumption order.
fn shapes(config: &BdhConfig) -> Vec<(u32, u32)> {
    let n = u32::try_from(config.latent()).expect("fits");
    let d = u32::try_from(config.hidden).expect("fits");
    let nh = u32::try_from(config.heads).expect("fits");
    let mut out = vec![(n, d); config.heads * 2];
    out.push((d, nh * n));
    out.push((u32::try_from(config.vocab).expect("fits"), d));
    out
}

fn to_spm(shapes: &[(u32, u32)], matrices: &[Vec<f32>]) -> Vec<u8> {
    let descriptors: Vec<OpDescriptor> = shapes
        .iter()
        .map(|(rows, cols)| OpDescriptor {
            rows: *rows,
            cols: *cols,
            group_size: 32,
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

fn fixture() -> (BdhConfig, Vec<u8>) {
    let config = config();
    let shapes = shapes(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 1, *r as usize * *c as usize))
        .collect();
    (config, to_spm(&shapes, &matrices))
}

#[test]
fn a_forward_pass_rewinds_once_per_level_and_never_seeks() {
    // The rotating-store property. BDH's loop body carries no layer
    // index, so one parameter set is swept n_layer times -- and the
    // only backward motion allowed is a rewind BETWEEN levels.
    let (config, bytes) = fixture();
    let positions = 5;
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let mut level = Level::new(&config, positions);
    let mut state = draw(999, positions * config.hidden);
    let mut logits = vec![0.0; positions * config.vocab];

    let rewinds =
        forward(&mut groups, &config, &mut state, &mut level, &mut logits).expect("forward");

    assert_eq!(rewinds, config.n_layer - 1, "one rewind between levels");
    assert!(state.iter().all(|v| v.is_finite()), "state went non-finite");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits went non-finite"
    );
    assert!(
        logits.iter().any(|v| *v != 0.0),
        "lm_head produced nothing -- it was not reached by reading on"
    );
}

#[test]
fn lm_head_sits_after_the_rotating_region() {
    // The structural claim layouts/bdh.order rests on: after the last
    // level's decoder sweep the cursor is exactly at lm_head, so the
    // logits need no rewind. If lm_head were inside the rotating
    // region, the final sweep would consume it as a weight matrix and
    // the shapes would not line up.
    let (config, bytes) = fixture();
    let groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    assert_eq!(
        groups.descriptors.len(),
        config.rotating_streams() + 1,
        "nine rotating streams plus lm_head"
    );
    let last = groups.descriptors.last().expect("lm_head");
    assert_eq!(
        (last.rows as usize, last.cols as usize),
        (config.vocab, config.hidden),
        "the final stream must be lm_head, not a rotating matrix"
    );
}

#[test]
fn the_sparse_latent_overtakes_the_whole_weight_set() {
    // THE FINDING OF THIS RUNG, asserted so it cannot be quietly lost.
    //
    // docs/plan.md section 3 keeps activations resident on the grounds
    // that they are kilobytes while weights are megabytes. BDH's
    // sparse latent is heads * positions * latent floats: it grows
    // linearly in sequence length while the weight set does not grow
    // at all, so the asymmetry does not merely weaken -- it inverts.
    let config = BdhConfig::default();
    let (small, weights) = config.budget(16);
    assert!(
        small * 20 < weights,
        "at 16 positions activations ({small} B) should still be a small \
         fraction of the weights ({weights} B)"
    );

    let (large, _) = config.budget(512);
    assert!(
        large > weights,
        "at 512 positions activations ({large} B) should EXCEED the whole \
         weight set ({weights} B) -- if this stops holding, either the \
         shape changed or a buffer was silently added or removed"
    );

    // And the growth really is linear in positions, which is what
    // makes the crossover inevitable rather than incidental.
    let (a, _) = config.budget(128);
    let (b, _) = config.budget(256);
    assert!(
        b > 2 * a - a / 10 && b < 2 * a + a / 10,
        "doubling positions should roughly double activations: {a} -> {b}"
    );
}
