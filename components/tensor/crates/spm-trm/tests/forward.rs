//! A whole forward pass, streamed, with the rotating region rewound
//! between every `L_level` call.
//!
//! Hermetic: TRM-shaped synthetic weights at a reduced width, not the
//! 27 MB checkpoint. The shapes and the recursion are what is under
//! test, and both are properties of the architecture rather than of
//! the trained values.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::{Layer, TrmConfig, forward};
use spm_walk::Cursor;

/// A narrow TRM: same structure, small enough to run in a test.
fn config() -> TrmConfig {
    TrmConfig {
        hidden: 16,
        heads: 2,
        expansion: 2,
        ..TrmConfig::default()
    }
}

/// Deterministic weights, small and exactly representable.
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

/// The eight rotating matrices, in consumption order.
fn rotating(config: &TrmConfig) -> Vec<(u32, u32)> {
    let w = u32::try_from(config.hidden).expect("fits");
    let inter = u32::try_from(config.intermediate()).expect("fits");
    (0..config.layers)
        .flat_map(|_| [(w * 3, w), (w, w), (inter * 2, w), (w, inter)])
        .collect()
}

/// Frames matrices into a `.spm` in the order given.
fn to_spm(shapes: &[(u32, u32)], matrices: &[Vec<f32>]) -> Vec<u8> {
    let descriptors: Vec<OpDescriptor> = shapes
        .iter()
        .map(|(rows, cols)| OpDescriptor {
            rows: *rows,
            cols: *cols,
            group_size: 64,
            encoding: Encoding::F32,
            lane_count: 1,
        })
        .collect();
    let mut writer = SpmWriter::new(descriptors.clone());
    let mut cursor = Cursor::new(&descriptors);
    for matrix in matrices {
        let mut at = 0usize;
        while let Some(count) = cursor.group_len(&descriptors) {
            let count = count as usize;
            let mut bytes = vec![0u8; dense_len(count)];
            encode_into(&matrix[at..at + count], &mut bytes).expect("encode");
            writer.write_raw_group(1.0, &bytes, count).expect("write");
            cursor.advance(&descriptors);
            at += count;
            if at >= matrix.len() {
                break;
            }
        }
    }
    writer.finish().expect("finish")
}

#[test]
fn a_forward_pass_rewinds_once_per_level_call() {
    // The property that makes this a rotating store. 3 * (4 + 1) = 15
    // L_level calls, so 14 rewinds -- one before each call after the
    // first, and never inside one.
    let config = config();
    let shapes = rotating(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 1, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, &matrices);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");

    let positions = 4;
    let mut state = draw(999, positions * config.hidden);
    let mut layers: Vec<Layer> = (0..config.layers)
        .map(|_| Layer::new(&config, positions))
        .collect();
    let report = forward(&mut groups, &config, &mut state, &mut layers).expect("forward");

    assert_eq!(report.calls, 15, "3 * (4 + 1) L_level calls");
    assert_eq!(report.rewinds, 14, "one rewind between consecutive calls");
    assert_eq!(report.positions, positions);
    // Every call re-reads the whole rotating region.
    assert_eq!(report.weights_read, report.weights_distinct * 15);
    assert!(state.iter().all(|v| v.is_finite()), "state went non-finite");
}

#[test]
fn scan_productivity_counts_recursion_as_well_as_batch() {
    // Ps as spm-stream-metrics defines it sees batch reuse only. For a
    // recursive model that under-reports by the number of L_level
    // calls -- 15x here. This is the honest number for a model that
    // rotates its weights.
    let config = config();
    let shapes = rotating(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 1, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, &matrices);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");

    let positions = 8;
    let mut state = draw(7, positions * config.hidden);
    let mut layers: Vec<Layer> = (0..config.layers)
        .map(|_| Layer::new(&config, positions))
        .collect();
    let report = forward(&mut groups, &config, &mut state, &mut layers).expect("forward");

    // positions * calls: 8 batch positions x 15 sweeps.
    let ps = report.scan_productivity().expect("Ps");
    let want = f64::from(u32::try_from(positions * 15).expect("fits"));
    assert!((ps - want).abs() < 1e-9, "Ps was {ps}, expected {want}");
    // A batch-only view would have said 8 -- 15x less.
    assert!(ps > 100.0, "recursion reuse must dominate batch reuse here");
}

#[test]
fn the_stream_is_consumed_exactly_and_rewinds_land_on_stream_zero() {
    // If a rewind landed on the header rather than the first group,
    // the next sweep would read metadata as weights and the state
    // would go non-finite almost immediately.
    let config = config();
    let shapes = rotating(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 100, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, &matrices);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let positions = 2;
    let mut state = draw(5, positions * config.hidden);
    let before = state.clone();
    let mut layers: Vec<Layer> = (0..config.layers)
        .map(|_| Layer::new(&config, positions))
        .collect();
    forward(&mut groups, &config, &mut state, &mut layers).expect("forward");
    assert!(state.iter().all(|v| v.is_finite()));
    assert_ne!(state, before, "the forward pass must actually change state");
}

#[test]
fn the_config_reproduces_the_real_checkpoints_shapes() {
    // The test that would have caught the intermediate-width bug, and
    // the reason it is written against PUBLISHED numbers rather than
    // generated ones. The earlier tests derived their shapes from the
    // same formula they were checking, so they agreed with a wrong
    // answer perfectly.
    //
    // yagizdevre/trm-maze-30x30, from its state dict:
    //
    //   self_attn.qkv_proj   (1536, 512)
    //   self_attn.o_proj     ( 512, 512)
    //   mlp.gate_up_proj     (3072, 512)
    //   mlp.down_proj        ( 512, 1536)
    let config = TrmConfig::default();
    assert_eq!(config.hidden, 512);
    assert_eq!(config.head_dim(), 64, "512 / 8 heads");

    // round(4 * 512 * 2/3) = 1365, aligned up to a multiple of 256.
    assert_eq!(config.intermediate(), 1536, "NOT hidden * expansion = 2048");

    let width = config.hidden;
    let inter = config.intermediate();
    assert_eq!((width * 3, width), (1536, 512), "qkv_proj");
    assert_eq!((width, width), (512, 512), "o_proj");
    assert_eq!((inter * 2, width), (3072, 512), "gate_up_proj");
    assert_eq!((width, inter), (512, 1536), "down_proj");
}

#[test]
fn a_forward_pass_reads_the_checkpoints_weight_count() {
    // 6,815,744 rotating weights, from the real state dict. If the
    // shape formulas drift, this catches it without a download.
    let config = TrmConfig::default();
    let (width, inter) = (config.hidden, config.intermediate());
    let per_layer = width * 3 * width + width * width + inter * 2 * width + width * inter;
    assert_eq!(per_layer * config.layers, 6_815_744);
}
