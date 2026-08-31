//! Loading the resident tensors and the greedy reference.
//!
//! Example scaffolding. Lives beside the example rather than in the
//! crate because none of it is part of the serving engine: it reads
//! what `scripts/extract-checkpoint` and `greedy.py` produced.

use spm_smol::SmolConfig;

/// Everything resident: the tied embedding table and the norms.
pub struct Fixture {
    pub embed: Vec<f32>,
    pub norms: Vec<(Vec<f32>, Vec<f32>)>,
    pub final_norm: Vec<f32>,
}

/// The greedy reference from `transformers`.
pub struct Reference {
    pub prompts: Vec<Vec<u32>>,
    pub produced: Vec<Vec<u32>>,
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One resident tensor, widened if the manifest says it is bf16.
fn blob(dir: &str, manifest: &str, name: &str) -> Vec<f32> {
    for line in manifest.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 4 && fields[0] == name {
            let path = format!("{dir}/{}", fields[3]);
            if fields[2] == "bf16" {
                let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
                let mut out = vec![0.0f32; raw.len() / 2];
                spm_codec_bf16::decode_into(&raw, &mut out).expect("bf16");
                return out;
            }
            return read_f32(&path);
        }
    }
    panic!("{name} not in manifest");
}

/// Undoes the extractor's transpose: the embedding is read by row.
fn to_row_major(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = src[col * rows + row];
        }
    }
    out
}

/// Reads every resident tensor from the extractor's output.
pub fn load(extracted: &str, config: &SmolConfig) -> Fixture {
    let manifest = std::fs::read_to_string(format!("{extracted}/manifest.tsv")).expect("manifest");
    let embed = to_row_major(
        &blob(extracted, &manifest, "model.embed_tokens.weight"),
        config.vocab,
        config.hidden,
    );
    let final_norm = blob(extracted, &manifest, "model.norm.weight");
    let norms = (0..config.layers)
        .map(|i| {
            (
                blob(
                    extracted,
                    &manifest,
                    &format!("model.layers.{i}.input_layernorm.weight"),
                ),
                blob(
                    extracted,
                    &manifest,
                    &format!("model.layers.{i}.post_attention_layernorm.weight"),
                ),
            )
        })
        .collect();
    Fixture {
        embed,
        norms,
        final_norm,
    }
}

/// Parses `greedy.tsv`: alternating prompt and produced rows.
pub fn greedy(dir: &str) -> Reference {
    let text = std::fs::read_to_string(format!("{dir}/greedy.tsv")).expect("greedy.tsv");
    let mut reference = Reference {
        prompts: Vec::new(),
        produced: Vec::new(),
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let Some((tag, values)) = line.split_once('\t') else {
            continue;
        };
        let row: Vec<u32> = values
            .split(',')
            .map(|v| v.trim().parse().expect("token id"))
            .collect();
        match tag {
            "prompt" => reference.prompts.push(row),
            "produced" => reference.produced.push(row),
            _ => {}
        }
    }
    reference
}
