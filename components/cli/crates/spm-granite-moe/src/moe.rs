use crate::{
    math::{softmax, top_k},
    weights,
};
use spm_gguf::Content;
use std::{collections::BTreeSet, env, path::Path};

const WIDTH: usize = 1024;
const FF: usize = 512;
const USED: usize = 8;

pub struct Trace {
    pub logits: Vec<Vec<f32>>,
    pub routes: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f32>>,
    pub contributions: Vec<Vec<Vec<f32>>>,
    pub output: Vec<Vec<f32>>,
}

fn route(logits: Vec<Vec<f32>>) -> Trace {
    let mut routes = Vec::new();
    let mut route_weights = Vec::new();
    for token in &logits {
        let probabilities = softmax(token);
        let selected = top_k(&probabilities, USED);
        let sum: f32 = selected.iter().map(|item| item.1).sum();
        routes.push(selected.iter().map(|item| item.0).collect());
        route_weights.push(selected.iter().map(|item| item.1 / sum).collect());
    }
    let count = logits.len();
    Trace {
        logits,
        routes,
        weights: route_weights,
        contributions: vec![vec![Vec::new(); USED]; count],
        output: vec![vec![0.0; WIDTH]; count],
    }
}

fn expert_tokens(trace: &Trace, expert: usize) -> Vec<usize> {
    trace
        .routes
        .iter()
        .enumerate()
        .filter_map(|(token, ids)| ids.contains(&expert).then_some(token))
        .collect()
}

fn activate(gate: &[Vec<f32>], up: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            gate.iter()
                .zip(up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect()
        })
        .collect()
}

fn accumulate(
    trace: &mut Trace,
    expert: usize,
    tokens: &[usize],
    down: Vec<Vec<f32>>,
) -> Result<(), String> {
    for (&token, value) in tokens.iter().zip(down) {
        let slot = trace.routes[token]
            .iter()
            .position(|&id| id == expert)
            .ok_or("missing expert slot")?;
        trace.contributions[token][slot].clone_from(&value);
        for (sum, lane) in trace.output[token].iter_mut().zip(value) {
            *sum += lane * trace.weights[token][slot];
        }
    }
    Ok(())
}

fn process_expert(
    path: &Path,
    content: &Content,
    layer: usize,
    expert: usize,
    input: &[Vec<f32>],
    trace: &mut Trace,
) -> Result<(), String> {
    let tokens = expert_tokens(trace, expert);
    let selected: Vec<_> = tokens
        .iter()
        .map(|&token| input[token].as_slice())
        .collect();
    let prefix = format!("blk.{layer}");
    let up = weights::project_expert(
        path,
        content,
        &format!("{prefix}.ffn_up_exps.weight"),
        WIDTH,
        FF,
        expert,
        &selected,
    )?;
    let gate = weights::project_expert(
        path,
        content,
        &format!("{prefix}.ffn_gate_exps.weight"),
        WIDTH,
        FF,
        expert,
        &selected,
    )?;
    let activated = activate(&gate, up);
    let inputs: Vec<_> = activated.iter().map(Vec::as_slice).collect();
    let down = weights::project_expert(
        path,
        content,
        &format!("{prefix}.ffn_down_exps.weight"),
        FF,
        WIDTH,
        expert,
        &inputs,
    )?;
    accumulate(trace, expert, &tokens, down)
}

pub fn run(
    path: &Path,
    content: &Content,
    input: &[Vec<f32>],
    layer: usize,
) -> Result<Trace, String> {
    let logits = weights::project_batch(
        path,
        content,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        WIDTH,
        input,
    )?;
    let mut trace = route(logits);
    let experts: BTreeSet<_> = trace.routes.iter().flatten().copied().collect();
    if layer == 0 && env::var_os("SPM_GRANITE_BATCH_REPORT").is_some() {
        let assignments = input.len() * USED;
        let heap_payload = assignments * (size_of::<usize>() + size_of::<f32>());
        let reuse = f32::from(u16::try_from(assignments).expect("assignments fit u16"))
            / f32::from(u16::try_from(experts.len()).expect("experts fit u16"));
        let sweep_reuse =
            f32::from(u16::try_from(assignments).expect("assignments fit u16")) / 32.0;
        println!(
            "expert_batch tokens={} assignments={assignments} unique={} selected_reuse={reuse:.6} full_sweep_reuse={sweep_reuse:.6} compact_state={} heap_payload={heap_payload} break_even_tokens=4",
            input.len(),
            experts.len(),
            assignments * 5
        );
    }
    for expert in experts {
        process_expert(path, content, layer, expert, input, &mut trace)?;
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_order_is_expert_id_order() {
        let trace = route(vec![
            (0_u16..32).map(|value| f32::from(value % 9)).collect(),
        ]);
        let experts: Vec<_> = trace
            .routes
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert!(experts.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
