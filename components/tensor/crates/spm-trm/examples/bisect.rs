//! Compare each stage of a block against the torch reference, to find
//! where a disagreement starts rather than guess at it.
//!
//!     cargo run -p spm-trm --example bisect -- <dir> <spm>

use spm_linear::streamed;
use spm_ops::{multi_head, residual_norm, swiglu_batch};
use spm_stream_groups::GroupStream;
use spm_stream_mem::MemoryWeightStream;
use spm_trm::TrmConfig;

fn read(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn report(label: &str, got: &[f32], want: &[f32]) {
    let (mut max, mut dot, mut ga, mut wa) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        max = max.max((g - w).abs());
        dot += f64::from(*g) * f64::from(*w);
        ga += f64::from(*g) * f64::from(*g);
        wa += f64::from(*w) * f64::from(*w);
    }
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-30);
    println!(
        "{label:<10} len {:>6}  max {:>10.3e}  rel {:>9.3e}  cos {:.10}",
        got.len(),
        max,
        max / scale,
        dot / (ga.sqrt() * wa.sqrt())
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("dir");
    let spm = args.next().expect("spm");
    let config = TrmConfig::default();
    let x = read(&format!("{dir}/input.f32"));
    let positions = x.len() / config.hidden;
    let (width, inter) = (config.hidden, config.intermediate());

    let mut groups = GroupStream::open(MemoryWeightStream::new(std::fs::read(&spm).expect("spm")))
        .expect("open");

    let mut qkv = vec![0.0f32; positions * width * 3];
    streamed(&mut groups, (width * 3, width), (&x, positions), &mut qkv).expect("qkv");
    report("s1 qkv", &qkv, &read(&format!("{dir}/s1_qkv.f32")));

    let mut attn = vec![0.0f32; positions * width];
    multi_head(
        &qkv,
        (config.heads, config.head_dim(), positions),
        config.rope_base,
        &mut attn,
    );
    report("s2 attn", &attn, &read(&format!("{dir}/s2_attn.f32")));

    let mut proj = vec![0.0f32; positions * width];
    streamed(&mut groups, (width, width), (&attn, positions), &mut proj).expect("o");
    let mut h1 = x.clone();
    residual_norm(&mut h1, &proj, config.eps, width);
    report("s3 h1", &h1, &read(&format!("{dir}/s3_h1.f32")));

    let mut gu = vec![0.0f32; positions * inter * 2];
    streamed(&mut groups, (inter * 2, width), (&h1, positions), &mut gu).expect("gate_up");
    let mut folded = vec![0.0f32; positions * inter];
    swiglu_batch(&gu, &mut folded, inter, positions);
    let mut down = vec![0.0f32; positions * width];
    streamed(&mut groups, (width, inter), (&folded, positions), &mut down).expect("down");
    let mut h2 = h1.clone();
    residual_norm(&mut h2, &down, config.eps, width);
    report("s4 out", &h2, &read(&format!("{dir}/expected.f32")));
}
