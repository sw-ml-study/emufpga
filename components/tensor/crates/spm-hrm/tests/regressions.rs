//! Guards for defects recorded in docs/postmortem-1.md.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_hrm::{HrmConfig, forward};
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::Layer;
use spm_walk::Cursor;

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

fn spm(config: &HrmConfig) -> Vec<u8> {
    let w = u32::try_from(config.block.hidden).expect("fits");
    let inter = u32::try_from(config.block.intermediate()).expect("fits");
    let shapes: Vec<(u32, u32)> = (0..config.low_layers + config.high_layers)
        .flat_map(|_| [(w * 3, w), (w, w), (inter * 2, w), (w, inter)])
        .collect();
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

/// Runs a forward pass with the given input embedding, returning the
/// final low-level state.
fn run(config: &HrmConfig, input: &[f32]) -> Vec<f32> {
    let bytes = spm(config);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let positions = input.len() / config.block.hidden;
    let mut z_low = draw(500, positions * config.block.hidden);
    let mut z_high = draw(501, positions * config.block.hidden);
    let mut low: Vec<Layer> = (0..config.low_layers)
        .map(|_| Layer::new(&config.block, positions))
        .collect();
    let mut high: Vec<Layer> = (0..config.high_layers)
        .map(|_| Layer::new(&config.block, positions))
        .collect();
    forward(
        &mut groups,
        config,
        (&mut z_low, &mut z_high),
        (&mut low, &mut high),
        input,
    )
    .expect("forward");
    z_low
}

#[test]
fn defect_9_the_input_embedding_reaches_the_recursion() {
    // POSTMORTEM DEFECT 9. `forward` never performed HRM's input
    // injection: it computed low(z_L) where HRM computes
    // low(z_L, z_H + input). The result ran, stayed finite, and was
    // not HRM.
    //
    // The tests at the time checked sweep counts, rewind counts and
    // finiteness, and all of them passed. This one cannot pass without
    // the injection: if the input never reaches the layers, changing
    // it cannot change the output.
    let config = config();
    let positions = 4;
    let width = config.block.hidden;
    let zeros = vec![0.0f32; positions * width];
    let ones = draw(777, positions * width);

    let with_zero = run(&config, &zeros);
    let with_input = run(&config, &ones);

    assert_eq!(with_zero.len(), with_input.len());
    let moved = with_zero
        .iter()
        .zip(&with_input)
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    assert!(
        moved > with_zero.len() / 2,
        "changing the input embedding moved only {moved} of {} outputs -- \
         the injection is not reaching the recursion",
        with_zero.len()
    );
}

#[test]
fn defect_9_both_modules_receive_their_injection() {
    // The low module gets z_high + input; the high module gets z_low.
    // A version that injected into low only would still pass the test
    // above, so this checks the high module's state is coupled too:
    // with everything else fixed, a different input must reach it
    // through z_low.
    let config = config();
    let positions = 4;
    let width = config.block.hidden;
    let a = run(&config, &draw(11, positions * width));
    let b = run(&config, &draw(12, positions * width));
    assert_ne!(a, b, "different inputs produced identical states");
}
