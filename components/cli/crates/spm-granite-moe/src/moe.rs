use crate::{
    math::{softmax, top_k},
    shape::MoeShape,
    weights,
};
use spm_gguf::Content;
use std::{collections::BTreeSet, env, path::Path};

pub struct Trace {
    pub logits: Vec<Vec<f32>>,
    pub routes: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f32>>,
    pub contributions: Vec<Vec<Vec<f32>>>,
    pub output: Vec<Vec<f32>>,
}

fn route(logits: Vec<Vec<f32>>, shape: MoeShape) -> Trace {
    let mut routes = Vec::new();
    let mut route_weights = Vec::new();
    for token in &logits {
        let probabilities = softmax(token);
        let selected = top_k(&probabilities, shape.used);
        let sum: f32 = selected.iter().map(|item| item.1).sum();
        routes.push(selected.iter().map(|item| item.0).collect());
        route_weights.push(selected.iter().map(|item| item.1 / sum).collect());
    }
    let count = logits.len();
    Trace {
        logits,
        routes,
        weights: route_weights,
        contributions: vec![vec![Vec::new(); shape.used]; count],
        output: vec![vec![0.0; shape.width]; count],
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
    shape: MoeShape,
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
        shape.width,
        shape.ff,
        expert,
        &selected,
    )?;
    let gate = weights::project_expert(
        path,
        content,
        &format!("{prefix}.ffn_gate_exps.weight"),
        shape.width,
        shape.ff,
        expert,
        &selected,
    )?;
    let activated = activate(&gate, up);
    let inputs: Vec<_> = activated.iter().map(Vec::as_slice).collect();
    let down = weights::project_expert(
        path,
        content,
        &format!("{prefix}.ffn_down_exps.weight"),
        shape.ff,
        shape.width,
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
    shape: MoeShape,
) -> Result<Trace, String> {
    shape.validate()?;
    let logits = weights::project_batch(
        path,
        content,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        shape.width,
        input,
    )?;
    let mut trace = route(logits, shape);
    let experts: BTreeSet<_> = trace.routes.iter().flatten().copied().collect();
    if layer == 0 && env::var_os("SPM_GRANITE_BATCH_REPORT").is_some() {
        let assignments = input.len() * shape.used;
        let heap_payload = assignments * (size_of::<usize>() + size_of::<f32>());
        let reuse = f32::from(u16::try_from(assignments).expect("assignments fit u16"))
            / f32::from(u16::try_from(experts.len()).expect("experts fit u16"));
        let sweep_reuse = f32::from(u16::try_from(assignments).expect("assignments fit u16"))
            / f32::from(u16::try_from(shape.experts).expect("experts fit u16"));
        println!(
            "expert_batch tokens={} assignments={assignments} unique={} selected_reuse={reuse:.6} full_sweep_reuse={sweep_reuse:.6} compact_state={} heap_payload={heap_payload} break_even_tokens=4",
            input.len(),
            experts.len(),
            assignments * 5
        );
    }
    for expert in experts {
        process_expert(path, content, layer, expert, shape, input, &mut trace)?;
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_order_is_expert_id_order() {
        let trace = route(
            vec![(0_u16..32).map(|value| f32::from(value % 9)).collect()],
            crate::shape::GRANITE,
        );
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

    #[test]
    fn olmoe_route_has_eight_of_sixty_four_experts_and_2048_outputs() {
        let trace = route(
            vec![(0_u16..64).map(f32::from).collect()],
            crate::shape::OLMOE,
        );
        assert_eq!(trace.routes[0].len(), 8);
        assert_eq!(trace.output[0].len(), 2048);
        assert_eq!(trace.routes[0], [63, 62, 61, 60, 59, 58, 57, 56]);
    }
}
