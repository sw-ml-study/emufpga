use crate::{
    math::{add_scaled, softmax},
    weights,
};
use spm_gguf::Content;
use std::path::Path;

const WIDTH: usize = 1024;
const HEAD: usize = 64;
const Q_HEADS: usize = 16;

fn rope(vector: &mut [f32], position: usize) {
    for head in vector.chunks_exact_mut(HEAD) {
        for pair in (0..HEAD).step_by(2) {
            let exponent = f32::from(u16::try_from(pair).expect("head lane fits u16"))
                / f32::from(u16::try_from(HEAD).expect("head width fits u16"));
            let position = f32::from(u16::try_from(position).expect("position fits u16"));
            let angle = position / 1_500_000_f32.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            (head[pair], head[pair + 1]) = (
                head[pair] * cos - head[pair + 1] * sin,
                head[pair] * sin + head[pair + 1] * cos,
            );
        }
    }
}

fn attend(queries: &[Vec<f32>], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<Vec<f32>> {
    queries
        .iter()
        .enumerate()
        .map(|(position, query)| {
            (0..Q_HEADS)
                .flat_map(|q_head| {
                    let kv_head = q_head / 2;
                    let q = &query[q_head * HEAD..(q_head + 1) * HEAD];
                    let scores: Vec<_> = keys[..=position]
                        .iter()
                        .map(|key| {
                            q.iter()
                                .zip(&key[kv_head * HEAD..(kv_head + 1) * HEAD])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * 0.015_625
                        })
                        .collect();
                    let scores = softmax(&scores);
                    (0..HEAD)
                        .map(|lane| {
                            values[..=position]
                                .iter()
                                .zip(&scores)
                                .map(|(value, score)| value[kv_head * HEAD + lane] * score)
                                .sum::<f32>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .collect()
}

pub fn run(
    path: &Path,
    content: &Content,
    hidden: &[Vec<f32>],
    normalized: &[Vec<f32>],
    layer: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let prefix = format!("blk.{layer}");
    let mut queries = weights::project_batch(
        path,
        content,
        &format!("{prefix}.attn_q.weight"),
        WIDTH,
        normalized,
    )?;
    let mut keys = weights::project_batch(
        path,
        content,
        &format!("{prefix}.attn_k.weight"),
        WIDTH,
        normalized,
    )?;
    let values = weights::project_batch(
        path,
        content,
        &format!("{prefix}.attn_v.weight"),
        WIDTH,
        normalized,
    )?;
    queries
        .iter_mut()
        .enumerate()
        .for_each(|(position, value)| rope(value, position));
    keys.iter_mut()
        .enumerate()
        .for_each(|(position, value)| rope(value, position));
    let mixed = attend(&queries, &keys, &values);
    let output = weights::project_batch(
        path,
        content,
        &format!("{prefix}.attn_output.weight"),
        WIDTH,
        &mixed,
    )?;
    Ok(hidden
        .iter()
        .zip(output)
        .map(|(residual, value)| add_scaled(residual, &value, 0.22))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_rope_is_identity() {
        let mut values: Vec<_> = (0_u16..1024).map(f32::from).collect();
        let before = values.clone();
        rope(&mut values, 0);
        assert_eq!(values, before);
    }
}
