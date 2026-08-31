//! Cross-check streamed SmolLM2-135M against `transformers`.
//!
//!     cargo run --release -p spm-smol --example smol-xcheck \
//!         -- <dir> <spm> <extracted>
//!
//! `<dir>` holds what `reference.py` dumped, `<spm>` is the imported
//! model, `<extracted>` is the extractor's blob directory, which is
//! where the resident tensors come from.
//!
//! Layer by layer, because `transformers` hands back every hidden
//! state and a single end-to-end number would say nothing about where
//! a disagreement started. On TRM that distinction found three bugs.

use spm_smol::{Layer, Resident, SmolConfig, forward};
use spm_smol_ops::scaled_rms_norm;
use spm_stream_file::FileWeightStream;
use spm_stream_groups::GroupStream;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(path: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine and error relative to the stage's own scale.
fn report(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let (mut dot, mut ga, mut wa) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        dot += f64::from(*g) * f64::from(*w);
        ga += f64::from(*g) * f64::from(*g);
        wa += f64::from(*w) * f64::from(*w);
    }
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let cosine = dot / (ga.sqrt() * wa.sqrt());
    println!(
        "{label:<12} cosine {cosine:.12}  max {max_abs:.3e}  rel {:.3e}",
        max_abs / scale
    );
}

/// The resident tensors, read from the extractor's blobs by name.
///
/// The extractor writes blobs in manifest order; this looks them up by
/// the names the manifest carries so a layout change cannot silently
/// shift which tensor is which.
fn resident_blob(dir: &str, manifest: &str, name: &str) -> Vec<f32> {
    for line in manifest.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 4 && fields[0] == name {
            let path = format!("{dir}/{}", fields[3]);
            // The resident tensors carry the same encoding as the
            // streamed ones. Reading a bf16 blob as f32 would halve
            // the count and produce garbage, so consult the manifest
            // rather than assuming -- the whole point of this step.
            return match fields[2] {
                "bf16" => {
                    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
                    let mut out = vec![0.0f32; bytes.len() / 2];
                    spm_codec_bf16::decode_into(&bytes, &mut out).expect("bf16");
                    out
                }
                _ => read_f32(&path),
            };
        }
    }
    panic!("{name} not in manifest");
}

/// Undoes the extractor's transpose for a tensor used by ROW.
///
/// `scripts/extract-checkpoint` writes every 2-D tensor in column-major
/// stream order, `blob[k] = W[k % rows][k / rows]`. That is what a
/// streamed matmul wants. `embed_tokens` is not streamed: it is
/// gathered by token id, and the tied output projection also needs
/// row `v` contiguous, so both uses want row-major.
///
/// The same tensor therefore needs two different layouts depending on
/// how it is read, and the extractor can only commit to one. Recorded
/// in docs/results.md rather than papered over here.
fn to_row_major(blob: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = blob[col * rows + row];
        }
    }
    out
}

/// Reads every resident tensor: the embedding table and the norms.
///
/// Returned rather than loaded inline so `main` stays readable. The
/// embedding is transposed back to row-major here, which is the one
/// place that conversion belongs.
type Residents = (Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>);

fn load_resident(extracted: &str, config: &SmolConfig) -> Residents {
    let manifest = std::fs::read_to_string(format!("{extracted}/manifest.tsv")).expect("manifest");
    let embed = to_row_major(
        &resident_blob(extracted, &manifest, "model.embed_tokens.weight"),
        config.vocab,
        config.hidden,
    );
    let final_norm = resident_blob(extracted, &manifest, "model.norm.weight");
    let norms = (0..config.layers)
        .map(|i| {
            (
                resident_blob(
                    extracted,
                    &manifest,
                    &format!("model.layers.{i}.input_layernorm.weight"),
                ),
                resident_blob(
                    extracted,
                    &manifest,
                    &format!("model.layers.{i}.post_attention_layernorm.weight"),
                ),
            )
        })
        .collect();
    (embed, final_norm, norms)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: smol-xcheck <dir> <spm> <extracted>");
    let spm = args.next().expect("spm");
    let extracted = args.next().expect("extracted");
    let config = SmolConfig::default();

    let resident = load_resident(&extracted, &config);
    let (embed, final_norm, norms) = (&resident.0, &resident.1, &resident.2);
    let tokens = read_u32(&format!("{dir}/tokens.u32"));
    let positions = tokens.len();
    let d = config.hidden;

    // Layer by layer, against transformers' own hidden states.
    let mut groups = GroupStream::open(FileWeightStream::open(&spm).expect("open")).expect("spm");
    let mut layer = Layer::new(&config, positions);
    let mut state = vec![0.0f32; positions * d];
    for (position, token) in tokens.iter().enumerate() {
        let row = *token as usize * d;
        state[position * d..(position + 1) * d].copy_from_slice(&embed[row..row + d]);
    }
    report(
        "hidden_0",
        &state,
        &read_f32(&format!("{dir}/stage_hidden_0.f32")),
    );
    for (index, norm) in norms.iter().enumerate().take(config.layers) {
        let pair = (&norm.0[..], &norm.1[..]);
        layer
            .forward(&mut groups, &config, pair, &mut state)
            .expect("layer");
        if index < 2 {
            let want = read_f32(&format!("{dir}/stage_hidden_{}.f32", index + 1));
            report(&format!("hidden_{}", index + 1), &state, &want);
        }
    }
    // transformers applies model.norm to its LAST hidden state but not
    // to the intermediate ones, so hidden_30 is norm(layer_30(...)).
    // Verified against the checkpoint rather than assumed: comparing
    // the raw state here reports cosine 0.32 and looks like a bug.
    let mut normed = state.clone();
    scaled_rms_norm(&mut normed, final_norm, config.eps, d);
    let want = read_f32(&format!("{dir}/stage_hidden_{}.f32", config.layers));
    report(
        &format!("hidden_{} (normed)", config.layers),
        &normed,
        &want,
    );

    // And the driver, end to end.
    let mut groups = GroupStream::open(FileWeightStream::open(&spm).expect("open")).expect("spm");
    let mut layer = Layer::new(&config, positions);
    let mut driven = vec![0.0f32; positions * d];
    let mut logits = vec![0.0f32; positions * config.vocab];
    let resident = Resident {
        embed,
        norms,
        final_norm,
    };
    let io = (&tokens[..], &mut driven[..], &mut logits[..]);
    let started = std::time::Instant::now();
    forward(&mut groups, &config, &resident, &mut layer, io).expect("forward");
    let took = started.elapsed();

    println!();
    report(
        "logits",
        &logits,
        &read_f32(&format!("{dir}/stage_logits.f32")),
    );
    println!("streams swept  {} (no rewind)", config.streams());

    // Traffic, and the rate a store would have to sustain. Unlike
    // every earlier rung this is the model read ONCE -- there is no
    // rotating region to re-read, so the traffic is the streamed
    // weight set exactly.
    // Traffic from the DESCRIPTORS, not from a weight count times four.
    // Assuming f32 here would report bf16's traffic as double what it
    // is and hide the whole point of the encoding.
    let streamed: usize = groups
        .descriptors
        .iter()
        .take(config.streams())
        .map(|d| d.rows as usize * d.cols as usize)
        .sum();
    let bytes: usize = groups
        .descriptors
        .iter()
        .take(config.streams())
        .map(|d| d.encoding.bytes_for(d.rows as usize * d.cols as usize))
        .sum();
    let encoding = groups.descriptors[0].encoding;
    let seconds = took.as_secs_f64();
    println!("positions      {positions}");
    println!("encoding       {encoding:?}");
    println!("streamed       {streamed} weights, {bytes} bytes, read once");
    println!("forward        {took:.3?}");
    println!(
        "store demand   {:.1} MB/s at this batch",
        spm_stream_metrics::widen(u64::try_from(bytes).unwrap_or(u64::MAX)) / seconds / 1.0e6
    );
}
