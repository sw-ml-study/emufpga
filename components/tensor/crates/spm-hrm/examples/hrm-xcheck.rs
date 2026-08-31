//! Compare one low-level module sweep against the OFFICIAL
//! sapientinc/HRM `ReasoningModule`, on the real checkpoint's weights.
//!
//!     cargo run -p spm-hrm --example xcheck -- <dir> <spm>
//!
//! The official code runs with a CPU stand-in for flash-attn, which is
//! a performance kernel rather than a different algorithm -- its output
//! is standard scaled dot-product attention. The ported checkpoint's
//! tensors load into the official module with zero missing and zero
//! unexpected keys, which is itself evidence that the shapes and the
//! name mapping are right.

use spm_hrm::HrmConfig;
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::Layer;

fn read(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: xcheck <dir> <spm>");
    let spm = args.next().expect("usage: xcheck <dir> <spm>");
    let config = HrmConfig::default();

    let hidden = read(&format!("{dir}/hidden.f32"));
    let inject = read(&format!("{dir}/inject.f32"));
    let expected = read(&format!("{dir}/expected.f32"));
    let positions = hidden.len() / config.block.hidden;

    let mut groups = GroupStream::open(MemoryWeightStream::new(std::fs::read(&spm).expect("spm")))
        .expect("open");
    let mut layers: Vec<Layer> = (0..config.low_layers)
        .map(|_| Layer::new(&config.block, positions))
        .collect();

    // ReasoningModule.forward: hidden = hidden + injection, then layers.
    let mut state = hidden;
    for (slot, add) in state.iter_mut().zip(&inject) {
        *slot += add;
    }
    for layer in &mut layers {
        layer
            .forward(&mut groups, &config.block, &mut state)
            .expect("layer");
    }

    let (mut max, mut dot, mut ga, mut wa) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in state.iter().zip(&expected) {
        max = max.max((g - w).abs());
        dot += f64::from(*g) * f64::from(*w);
        ga += f64::from(*g) * f64::from(*g);
        wa += f64::from(*w) * f64::from(*w);
    }
    let scale = expected.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    println!("positions      {positions}");
    println!("layers         {}", config.low_layers);
    println!("output scale   {scale:.6}");
    println!("max abs error  {max:.3e}");
    println!("relative       {:.3e}", max / scale);
    println!("cosine         {:.12}", dot / (ga.sqrt() * wa.sqrt()));
}
