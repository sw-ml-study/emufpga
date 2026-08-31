//! The resident path must compute exactly what the streamed path
//! computes, or the comparison in docs/results.md is meaningless.
//!
//! A performance comparison between two implementations that produce
//! different answers is not a comparison. This is the precondition,
//! asserted rather than eyeballed, and it is bit-exact rather than
//! within-tolerance: both paths apply weights in the same order and
//! both use `mul_add`, so there is no rounding difference left for a
//! tolerance to absorb. Anything less than equality is a bug.
//!
//! Hermetic: TRM-shaped synthetic weights at a reduced width, like
//! `spm-trm`'s own tests. The 27 MB checkpoint is exercised by the
//! `trm-compare` example, whose result docs/results.md records.

use spm_codec_dense::{dense_len, encode_into};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::{Layer, TrmConfig};
use spm_trm_resident::{ResidentLayer, ResidentWeights};

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
    for (matrix, descriptor) in matrices.iter().zip(&descriptors) {
        let group = descriptor.group_size as usize;
        for chunk in matrix.chunks(group) {
            let mut bytes = vec![0u8; dense_len(chunk.len())];
            encode_into(chunk, &mut bytes).expect("encode");
            writer
                .write_raw_group(1.0, &bytes, chunk.len())
                .expect("write");
        }
    }
    writer.finish().expect("finish")
}

/// Builds one fixture: the `.spm` bytes and the shared input state.
fn fixture(positions: usize) -> (TrmConfig, Vec<u8>, Vec<f32>) {
    let config = config();
    let shapes = rotating(&config);
    let matrices: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (r, c))| draw(i as u64 + 1, *r as usize * *c as usize))
        .collect();
    let bytes = to_spm(&shapes, &matrices);
    let state = draw(999, positions * config.hidden);
    (config, bytes, state)
}

#[test]
fn resident_and_streamed_forward_passes_agree_bit_for_bit() {
    let positions = 4;
    let (config, bytes, input) = fixture(positions);

    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.clone())).expect("open");
    let mut streamed_layers: Vec<Layer> = (0..config.layers)
        .map(|_| Layer::new(&config, positions))
        .collect();
    let mut streamed_state = input.clone();
    spm_trm::forward(
        &mut groups,
        &config,
        &mut streamed_state,
        &mut streamed_layers,
    )
    .expect("streamed forward");

    let mut load = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let weights = ResidentWeights::load(&mut load).expect("load");
    let mut resident_layers: Vec<ResidentLayer> = (0..config.layers)
        .map(|_| ResidentLayer::new(&config, positions))
        .collect();
    let mut resident_state = input;
    spm_trm_resident::forward(&weights, &config, &mut resident_state, &mut resident_layers)
        .expect("resident forward");

    // Bit patterns, not approximate equality: a difference of one ULP
    // would mean the two paths do not apply weights in the same order,
    // which is a defect in one of them rather than a tolerance to widen.
    let differing = streamed_state
        .iter()
        .zip(&resident_state)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} outputs differ; streamed {:?} vs resident {:?}",
        streamed_state.len(),
        &streamed_state[..4.min(streamed_state.len())],
        &resident_state[..4.min(resident_state.len())]
    );
    assert!(
        streamed_state.iter().all(|v| v.is_finite()),
        "state went non-finite -- equality would be vacuous"
    );
}

#[test]
fn the_resident_store_holds_every_parameter() {
    // The memory axis of the comparison, asserted rather than assumed.
    // Streaming holds one group; the resident path holds the model.
    let positions = 4;
    let (config, bytes, _) = fixture(positions);
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let one_group = groups.resident_parameter_bytes();
    let weights = ResidentWeights::load(&mut groups).expect("load");

    let shapes = rotating(&config);
    let expected: usize = shapes
        .iter()
        .map(|(r, c)| *r as usize * *c as usize)
        .sum::<usize>()
        * size_of::<f32>();
    assert_eq!(weights.parameter_bytes(), expected, "every weight held");
    assert!(
        one_group * 8 < weights.parameter_bytes(),
        "a group ({one_group} B) should be far smaller than the model \
         ({} B), or the residency claim is empty at this width",
        weights.parameter_bytes()
    );
}
