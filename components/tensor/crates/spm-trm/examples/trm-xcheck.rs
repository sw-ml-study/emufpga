//! Cross-check one TRM block against the `PyTorch` reference.
//!
//!     cargo run -p spm-trm --example xcheck -- <dir> <spm>
//!
//! `<dir>` holds `input.f32` and `expected.f32` produced by
//! `reference.py`; `<spm>` is the same weights framed by
//! `emufpga import`.
//!
//! An example rather than a test on purpose. It needs torch-generated
//! data and 13 MB of weights, and weights never enter this repository
//! -- so this is a manual verification whose result is recorded in
//! docs/results.md, exactly like the checkpoint import.
//!
//! **Tolerance, not bit-exactness, and the distinction matters.**
//! Bit-exactness is claimed only between this crate's own streamed and
//! resident paths, where the summation order is identical by
//! construction. Against torch it cannot hold: its GEMM and fused
//! attention accumulate in a different order, and the last bits will
//! differ. What this checks is that the *formulas* agree -- `RoPE`
//! base and rotation style, the qkv split, unmasked attention,
//! post-norm residuals, the `SwiGLU` gate/up order and the
//! intermediate width.

use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::{Layer, TrmConfig};

/// Reads a flat little-endian f32 file.
fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: xcheck <dir> <spm>");
    let spm = args.next().expect("usage: xcheck <dir> <spm>");

    let config = TrmConfig::default();
    let input = read_f32(&format!("{dir}/input.f32"));
    let expected = read_f32(&format!("{dir}/expected.f32"));
    let positions = input.len() / config.hidden;

    let mut groups = GroupStream::open(MemoryWeightStream::new(std::fs::read(&spm).expect("spm")))
        .expect("open");
    let mut layer = Layer::new(&config, positions);
    let mut state = input;
    layer
        .forward(&mut groups, &config, &mut state)
        .expect("block forward");

    let mut max_abs = 0.0f32;
    let mut sum_sq = (0.0f64, 0.0f64, 0.0f64);
    for (got, want) in state.iter().zip(&expected) {
        max_abs = max_abs.max((got - want).abs());
        sum_sq.0 += f64::from(*got) * f64::from(*want);
        sum_sq.1 += f64::from(*got) * f64::from(*got);
        sum_sq.2 += f64::from(*want) * f64::from(*want);
    }
    let cosine = sum_sq.0 / (sum_sq.1.sqrt() * sum_sq.2.sqrt());
    let scale = expected.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    println!("positions      {positions}");
    println!("hidden         {}", config.hidden);
    println!("intermediate   {}", config.intermediate());
    println!("output scale   {scale:.6}");
    println!("max abs error  {max_abs:.3e}");
    println!("relative       {:.3e}", max_abs / scale);
    println!("cosine         {cosine:.12}");
}
