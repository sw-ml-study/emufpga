use crate::{
    attention::{KvCache, causal_gqa, rope},
    compare,
    math::{add, print_stats, rms_norm},
    weights::{load, projection_batch, tensor},
};
use spm_gguf::{Content, decode_q6_k, read_tensor_range};
use std::{env, path::Path};

const WIDTH: usize = 4096;
const HEAD: usize = 128;
const FF: usize = 12288;

fn embedding(path: &Path, content: &Content, token: usize) -> Result<Vec<f32>, String> {
    let info = tensor(content, "token_embd.weight")?;
    let row_bytes = (WIDTH / 256) * 210;
    let start = token
        .checked_mul(row_bytes)
        .ok_or("token offset overflow")?;
    let offset = u64::try_from(start).map_err(|_| "token offset does not fit u64")?;
    let len = u64::try_from(row_bytes).map_err(|_| "row length does not fit u64")?;
    decode_q6_k(&read_tensor_range(path, info, offset, len, len)?)
}

fn attention(path: &Path, content: &Content, hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
    let norm = load(path, content, "blk.0.attn_norm.weight")?;
    let normalized: Vec<_> = hidden.iter().map(|token| rms_norm(token, &norm)).collect();
    let mut queries = projection_batch(path, content, "blk.0.attn_q.weight", WIDTH, &normalized)?;
    let mut keys = projection_batch(path, content, "blk.0.attn_k.weight", WIDTH, &normalized)?;
    let values = projection_batch(path, content, "blk.0.attn_v.weight", WIDTH, &normalized)?;
    let q_norm = load(path, content, "blk.0.attn_q_norm.weight")?;
    let k_norm = load(path, content, "blk.0.attn_k_norm.weight")?;
    for (position, query) in queries.iter_mut().enumerate() {
        for head in query.chunks_exact_mut(HEAD) {
            head.copy_from_slice(&rms_norm(head, &q_norm));
            rope(head, position);
        }
    }
    for (position, key) in keys.iter_mut().enumerate() {
        for head in key.chunks_exact_mut(HEAD) {
            head.copy_from_slice(&rms_norm(head, &k_norm));
            rope(head, position);
        }
    }
    let attended = causal_gqa(&queries, &KvCache { keys, values });
    let projected = projection_batch(path, content, "blk.0.attn_output.weight", WIDTH, &attended)?;
    Ok(hidden
        .iter()
        .zip(projected)
        .map(|(left, right)| add(left, &right))
        .collect())
}

fn feed_forward(
    path: &Path,
    content: &Content,
    hidden: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, String> {
    let norm = load(path, content, "blk.0.ffn_norm.weight")?;
    let normalized: Vec<_> = hidden.iter().map(|token| rms_norm(token, &norm)).collect();
    let gates = projection_batch(path, content, "blk.0.ffn_gate.weight", WIDTH, &normalized)?;
    let ups = projection_batch(path, content, "blk.0.ffn_up.weight", WIDTH, &normalized)?;
    let activated: Vec<Vec<f32>> = gates
        .iter()
        .zip(ups)
        .map(|(gate, up)| {
            gate.iter()
                .zip(up)
                .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
                .collect()
        })
        .collect();
    if activated.iter().any(|token| token.len() != FF) {
        return Err("unexpected feed-forward width".into());
    }
    let down = projection_batch(path, content, "blk.0.ffn_down.weight", FF, &activated)?;
    Ok(hidden
        .iter()
        .zip(down)
        .map(|(left, right)| add(left, &right))
        .collect())
}

fn check_stage(
    reference: Option<&Path>,
    name: &str,
    values: &[Vec<f32>],
    max_abs: f32,
    min_cosine: f64,
) -> Result<(), String> {
    for (position, value) in values.iter().enumerate() {
        print_stats(&format!("{name}[{position}]"), value);
        if let Some(directory) = reference {
            compare::check(directory, name, position, value, max_abs, min_cosine)?;
        }
    }
    Ok(())
}

pub fn run(path: &Path, tokens: &[usize]) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("at least one token is required".into());
    }
    let content = spm_gguf::read(path)?;
    let reference = env::var_os("SPM_QWEN_REFERENCE_DIR").map(std::path::PathBuf::from);
    let mut hidden: Vec<_> = tokens
        .iter()
        .map(|&token| embedding(path, &content, token))
        .collect::<Result<_, _>>()?;
    check_stage(
        reference.as_deref(),
        "inp_embd",
        &hidden,
        0.0,
        0.999_999_999,
    )?;
    hidden = attention(path, &content, &hidden)?;
    check_stage(reference.as_deref(), "ffn_inp-0", &hidden, 0.006, 0.99985)?;
    hidden = feed_forward(path, &content, &hidden)?;
    check_stage(reference.as_deref(), "l_out-0", &hidden, 0.06, 0.9997)
}
