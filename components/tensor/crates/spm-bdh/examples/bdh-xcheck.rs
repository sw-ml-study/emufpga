//! Cross-check streamed BDH against `pathwaycom/bdh`, stage by stage.
//!
//!     cargo run --release -p spm-bdh --example bdh-xcheck -- <dir> <spm>
//!
//! `<dir>` holds what `reference.py` dumped; `<spm>` is the same
//! weights framed by `emufpga import --order layouts/bdh.order`.
//!
//! **Stage by stage on purpose.** A single end-to-end number tells you
//! that something is wrong and nothing about where. On TRM, bisecting
//! this way put three separate bugs on one function each in a single
//! run; the end-to-end cosine had been 0.9993 and uninformative.
//!
//! Tolerance, not bit-exactness: torch's GEMM and this engine
//! accumulate in different orders. Bit-exactness is claimed only
//! between this repo's own streamed and resident paths.

use spm_bdh::{BdhConfig, Level, forward};
use spm_ops::layer_norm;
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;

/// Token ids, stored as u32 because they are integers -- reading them
/// as f32 would force a float-to-integer cast at the lookup.
fn read_u32(path: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine, max absolute error, and error relative to the output scale.
fn compare(got: &[f32], want: &[f32]) -> (f64, f32, f32) {
    let mut max_abs = 0.0f32;
    let (mut dot, mut ga, mut wa) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        dot += f64::from(*g) * f64::from(*w);
        ga += f64::from(*g) * f64::from(*g);
        wa += f64::from(*w) * f64::from(*w);
    }
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    (dot / (ga.sqrt() * wa.sqrt()), max_abs, max_abs / scale)
}

fn report(label: &str, got: &[f32], want: &[f32]) {
    let (cosine, max_abs, relative) = compare(got, want);
    println!("{label:<16} cosine {cosine:.12}  max {max_abs:.3e}  rel {relative:.3e}");
    assert_eq!(got.len(), want.len(), "{label}: length");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: bdh-xcheck <dir> <spm>");
    let spm = args.next().expect("usage: bdh-xcheck <dir> <spm>");
    let config = BdhConfig::default();

    let tokens = read_u32(&format!("{dir}/tokens.u32"));
    let embed = read_f32(&format!("{dir}/embed.f32"));
    let positions = tokens.len();

    // x = ln(embed(idx)). The lookup is a gather, which is why the
    // table stays resident -- see layouts/bdh.order.
    let mut state = vec![0.0f32; positions * config.hidden];
    for (position, token) in tokens.iter().enumerate() {
        let row = usize::try_from(*token).expect("token id") * config.hidden;
        state[position * config.hidden..(position + 1) * config.hidden]
            .copy_from_slice(&embed[row..row + config.hidden]);
    }
    layer_norm(&mut state, 1e-5, config.hidden);
    report(
        "x_embedded",
        &state,
        &read_f32(&format!("{dir}/stage_x_embedded.f32")),
    );

    let bytes = std::fs::read(&spm).expect("spm");
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes.clone())).expect("open");
    let mut level = Level::new(&config, positions);
    for pass in 0..config.n_layer {
        if pass > 0 {
            groups.rewind().expect("rewind");
        }
        level
            .forward(&mut groups, &config, &mut state)
            .expect("level");
        report(
            &format!("x_after_{pass}"),
            &state,
            &read_f32(&format!("{dir}/stage_x_after_{pass}.f32")),
        );
    }

    // The driver, end to end, must reach the same place.
    let mut groups = GroupStream::open(MemoryWeightStream::new(bytes)).expect("open");
    let mut level = Level::new(&config, positions);
    let mut driven = vec![0.0f32; positions * config.hidden];
    for (position, token) in tokens.iter().enumerate() {
        let row = usize::try_from(*token).expect("token id") * config.hidden;
        driven[position * config.hidden..(position + 1) * config.hidden]
            .copy_from_slice(&embed[row..row + config.hidden]);
    }
    layer_norm(&mut driven, 1e-5, config.hidden);
    let mut logits = vec![0.0f32; positions * config.vocab];
    let rewinds =
        forward(&mut groups, &config, &mut driven, &mut level, &mut logits).expect("forward");

    println!();
    report(
        "logits",
        &logits,
        &read_f32(&format!("{dir}/stage_logits.f32")),
    );
    println!("rewinds          {rewinds} (expect {})", config.n_layer - 1);
    println!("rotating streams {}", config.rotating_streams());
}
