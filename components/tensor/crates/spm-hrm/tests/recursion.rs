//! HRM's two-module recursion, and the layout it needs.
//!
//! The open question when this rung began was whether two modules
//! force two rotating regions -- which `rewind` alone could not serve,
//! since it returns to the start of the stream. These tests answer it.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_hrm::{HrmConfig, forward};
use spm_layout::{Encoding, OpDescriptor};
use spm_order::parse_order;
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::Layer;
use spm_walk::Cursor;

/// A narrow HRM: same structure, small enough for a test.
fn config() -> HrmConfig {
    let mut c = HrmConfig::default();
    c.block.hidden = 16;
    c.block.heads = 2;
    c
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

/// The 32 rotating shapes, low module then high, four per layer.
fn shapes(config: &HrmConfig) -> Vec<(u32, u32)> {
    let w = u32::try_from(config.block.hidden).expect("fits");
    let inter = u32::try_from(config.block.intermediate()).expect("fits");
    (0..config.low_layers + config.high_layers)
        .flat_map(|_| [(w * 3, w), (w, w), (inter * 2, w), (w, inter)])
        .collect()
}

fn to_spm(shapes: &[(u32, u32)]) -> Vec<u8> {
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
    for (index, (rows, cols)) in shapes.iter().enumerate() {
        let matrix = draw(index as u64 + 1, *rows as usize * *cols as usize);
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
fn two_modules_need_only_one_rotating_region() {
    // THE QUESTION THIS RUNG OPENED. A rewind returns to stream 0, so
    // if high-level weights needed reaching independently the layout
    // would need a seek and there is none. With low first, the last
    // low sweep of an outer cycle leaves the cursor exactly where high
    // begins, and high simply continues forward.
    let config = config();
    let bytes = to_spm(&shapes(&config));
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");

    let positions = 4;
    let width = config.block.hidden;
    let mut z_low = draw(900, positions * width);
    let mut z_high = draw(901, positions * width);
    let mut low: Vec<Layer> = (0..config.low_layers)
        .map(|_| Layer::new(&config.block, positions))
        .collect();
    let mut high: Vec<Layer> = (0..config.high_layers)
        .map(|_| Layer::new(&config.block, positions))
        .collect();

    let report = forward(
        &mut groups,
        &config,
        (&mut z_low, &mut z_high),
        (&mut low, &mut high),
    )
    .expect("forward");

    assert_eq!(report.low_sweeps, config.h_cycles * config.l_cycles);
    assert_eq!(report.high_sweeps, config.h_cycles);
    // One rewind before every low sweep except the very first, and
    // never before a high sweep.
    assert_eq!(report.rewinds, config.rewinds());
    assert_eq!(report.rewinds, config.h_cycles * config.l_cycles - 1);
    assert!(z_low.iter().all(|v| v.is_finite()));
    assert!(z_high.iter().all(|v| v.is_finite()));
}

#[test]
fn the_shipped_order_matches_the_published_checkpoint() {
    // Asserted against PUBLISHED numbers rather than anything this
    // codebase computed -- the rule docs/postmortem-1.md draws from
    // the SwiGLU width bug, where the tests generated their shapes
    // from the same wrong formula they were checking.
    //
    // zbloss/HRM-sudoku-extreme: 27,276,802 parameters, 39 tensors,
    // 32 weight matrices, intermediate_size 1536 stated in config.json.
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../layouts/hrm-sudoku-extreme.order"),
    )
    .expect("order file");
    let order = parse_order(&text).expect("parse");
    assert_eq!(
        order.rotating.len(),
        32,
        "4 matrices x 4 layers x 2 modules"
    );
    assert_eq!(order.resident.len(), 7);

    // Low module must come first, or the high sweep needs a seek back.
    for name in &order.rotating[..16] {
        assert!(name.contains("low_level_module"), "{name}");
    }
    for name in &order.rotating[16..] {
        assert!(name.contains("high_level_module"), "{name}");
    }
    // Within a layer, execution order, not the checkpoint's alphabetical.
    let parts = [
        "qkv_projection",
        "output_projection",
        "gate_up_projection",
        "down_projection",
    ];
    for (index, name) in order.rotating.iter().enumerate() {
        assert!(name.contains(parts[index % 4]), "stream {index}: {name}");
    }

    let config = HrmConfig::default();
    assert_eq!(config.block.intermediate(), 1536, "config.json states 1536");
    assert_eq!(config.rotating_streams(), 32);
    let (w, inter) = (config.block.hidden, config.block.intermediate());
    let per_layer = w * 3 * w + w * w + inter * 2 * w + w * inter;
    assert_eq!(per_layer * 8, 27_262_976, "published weight-matrix total");
}
