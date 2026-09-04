use crate::{
    attention::{KvCache, causal_gqa, normalize_rope},
    compare,
    math::{add, rms_norm, top_k},
    weights::{load, projection_batch, projection_stream, tensor},
};
use spm_gguf::{Content, decode_q6_k, read_tensor_range};
use std::{env, path::Path};

const WIDTH: usize = 4096;
const FF: usize = 12288;
const LAYERS: usize = 36;

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

fn embeddings(path: &Path, content: &Content, tokens: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    if tokens.is_empty() {
        return Err("at least one token is required".into());
    }
    tokens
        .iter()
        .map(|&token| embedding(path, content, token))
        .collect()
}

fn attention(
    path: &Path,
    content: &Content,
    hidden: &[Vec<f32>],
    layer: usize,
    cache: &mut KvCache,
) -> Result<Vec<Vec<f32>>, String> {
    let norm = load(path, content, &format!("blk.{layer}.attn_norm.weight"))?;
    let normalized: Vec<_> = hidden.iter().map(|token| rms_norm(token, &norm)).collect();
    let mut queries = projection_batch(
        path,
        content,
        &format!("blk.{layer}.attn_q.weight"),
        WIDTH,
        &normalized,
    )?;
    let mut keys = projection_batch(
        path,
        content,
        &format!("blk.{layer}.attn_k.weight"),
        WIDTH,
        &normalized,
    )?;
    let values = projection_batch(
        path,
        content,
        &format!("blk.{layer}.attn_v.weight"),
        WIDTH,
        &normalized,
    )?;
    let q_norm = load(path, content, &format!("blk.{layer}.attn_q_norm.weight"))?;
    let k_norm = load(path, content, &format!("blk.{layer}.attn_k_norm.weight"))?;
    normalize_rope(&mut queries, &q_norm);
    normalize_rope(&mut keys, &k_norm);
    *cache = KvCache { keys, values };
    let attended = causal_gqa(&queries, cache);
    let projected = projection_batch(
        path,
        content,
        &format!("blk.{layer}.attn_output.weight"),
        WIDTH,
        &attended,
    )?;
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
    layer: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let norm = load(path, content, &format!("blk.{layer}.ffn_norm.weight"))?;
    let normalized: Vec<_> = hidden.iter().map(|token| rms_norm(token, &norm)).collect();
    let gates = projection_batch(
        path,
        content,
        &format!("blk.{layer}.ffn_gate.weight"),
        WIDTH,
        &normalized,
    )?;
    let ups = projection_batch(
        path,
        content,
        &format!("blk.{layer}.ffn_up.weight"),
        WIDTH,
        &normalized,
    )?;
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
    let down = projection_batch(
        path,
        content,
        &format!("blk.{layer}.ffn_down.weight"),
        FF,
        &activated,
    )?;
    Ok(hidden
        .iter()
        .zip(down)
        .map(|(left, right)| add(left, &right))
        .collect())
}

fn block(
    path: &Path,
    content: &Content,
    hidden: &[Vec<f32>],
    layer: usize,
    cache: &mut KvCache,
) -> Result<Vec<Vec<f32>>, String> {
    feed_forward(
        path,
        content,
        &attention(path, content, hidden, layer, cache)?,
        layer,
    )
}

pub fn run(path: &Path, tokens: &[usize]) -> Result<(), String> {
    let content = spm_gguf::read(path)?;
    let reference = env::var_os("SPM_QWEN_REFERENCE_DIR").map(std::path::PathBuf::from);
    let mut hidden = embeddings(path, &content, tokens)?;
    compare::check_stage(
        reference.as_deref(),
        "inp_embd",
        &hidden,
        (0.0, 0.999_999_999),
    )?;
    let mut caches: Vec<_> = (0..LAYERS).map(|_| KvCache::default()).collect();
    for (layer, cache) in caches.iter_mut().enumerate() {
        hidden = block(path, &content, &hidden, layer, cache)?;
        let name = format!("l_out-{layer}");
        compare::check_stage(reference.as_deref(), &name, &hidden, (287.0, 0.997))?;
    }
    let norm = load(path, &content, "output_norm.weight")?;
    let final_hidden = rms_norm(hidden.last().ok_or("missing final token")?, &norm);
    let logits = projection_stream(path, &content, "output.weight", WIDTH, &final_hidden)?;
    let top = top_k(&logits, 10);
    let golden = env::var_os("SPM_QWEN_LOGITS_GOLDEN").map(std::path::PathBuf::from);
    compare::check_top(golden.as_deref(), &top, 0.5)
}
