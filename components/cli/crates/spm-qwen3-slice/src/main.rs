use spm_gguf::{Content, decode_q6_k, read_tensor_range};
use std::{env, path::Path};

mod math;
mod weights;
use math::{add, print_stats, rms_norm};
use weights::{load, projection, tensor};

const WIDTH: usize = 4096;
const HEAD: usize = 128;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const FF: usize = 12288;
fn embedding(path: &Path, content: &Content, token: usize) -> Result<Vec<f32>, String> {
    let embedding = tensor(content, "token_embd.weight")?;
    let row_bytes = (WIDTH / 256) * 210;
    let start = token
        .checked_mul(row_bytes)
        .ok_or("token offset overflow")?;
    let bytes = read_tensor_range(
        path,
        embedding,
        u64::try_from(start).map_err(|_| "token offset does not fit u64")?,
        u64::try_from(row_bytes).map_err(|_| "row length does not fit u64")?,
        u64::try_from(row_bytes).map_err(|_| "row length does not fit u64")?,
    )?;
    decode_q6_k(&bytes)
}

fn attention(path: &Path, content: &Content, hidden: &[f32]) -> Result<Vec<f32>, String> {
    let attn_norm = load(path, content, "blk.0.attn_norm.weight")?;
    let normalized = rms_norm(hidden, &attn_norm);
    let mut query = projection(path, content, "blk.0.attn_q.weight", WIDTH, &normalized)?;
    let mut key = projection(path, content, "blk.0.attn_k.weight", WIDTH, &normalized)?;
    let value = projection(path, content, "blk.0.attn_v.weight", WIDTH, &normalized)?;
    let q_norm = load(path, content, "blk.0.attn_q_norm.weight")?;
    let k_norm = load(path, content, "blk.0.attn_k_norm.weight")?;
    for head in query.chunks_exact_mut(HEAD) {
        head.copy_from_slice(&rms_norm(head, &q_norm));
    }
    for head in key.chunks_exact_mut(HEAD) {
        head.copy_from_slice(&rms_norm(head, &k_norm));
    }
    // At sequence position zero RoPE is the identity and a one-element causal
    // softmax is exactly one. Each group of four query heads therefore selects
    // its corresponding value head.
    let mut attended = Vec::with_capacity(WIDTH);
    for q_head in 0..Q_HEADS {
        let kv_head = q_head / (Q_HEADS / KV_HEADS);
        attended.extend_from_slice(&value[kv_head * HEAD..(kv_head + 1) * HEAD]);
    }
    let attention = projection(path, content, "blk.0.attn_output.weight", WIDTH, &attended)?;
    Ok(add(hidden, &attention))
}

fn feed_forward(path: &Path, content: &Content, hidden: &[f32]) -> Result<Vec<f32>, String> {
    let ffn_norm = load(path, content, "blk.0.ffn_norm.weight")?;
    let normalized = rms_norm(hidden, &ffn_norm);
    let gate = projection(path, content, "blk.0.ffn_gate.weight", WIDTH, &normalized)?;
    let up = projection(path, content, "blk.0.ffn_up.weight", WIDTH, &normalized)?;
    if gate.len() != FF || up.len() != FF {
        return Err("unexpected feed-forward width".into());
    }
    let activated: Vec<_> = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect();
    let down = projection(path, content, "blk.0.ffn_down.weight", FF, &activated)?;
    Ok(add(hidden, &down))
}

fn run(path: &Path, token: usize) -> Result<(), String> {
    let content = spm_gguf::read(path)?;
    let mut hidden = embedding(path, &content, token)?;
    print_stats("embedding", &hidden);
    hidden = attention(path, &content, &hidden)?;
    print_stats("post_attention", &hidden);
    hidden = feed_forward(path, &content, &hidden)?;
    print_stats("block_0", &hidden);
    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: spm-qwen3-slice MODEL.gguf TOKEN_ID");
        std::process::exit(2);
    }
    let token = args[2].parse().unwrap_or_else(|_| {
        eprintln!("invalid token ID");
        std::process::exit(2);
    });
    if let Err(error) = run(Path::new(&args[1]), token) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
