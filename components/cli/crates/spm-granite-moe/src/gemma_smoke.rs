use crate::{math, weights};
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_stream_file::FileWeightStream;
use spm_stream_groups::GroupStream;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    time::{Duration, Instant},
};

const WIDTH: usize = 2816;
const WIDTH_F32: f32 = 2816.0;
const FF: usize = 704;
const USED: usize = 8;
const Q5_VALUES: usize = 256;
const Q5_BYTES: usize = 176;
const Q8_VALUES: usize = 32;
const Q8_BYTES: usize = 34;

fn inputs(batch: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|token| {
            (0..WIDTH)
                .map(|lane| {
                    let value = u16::try_from((token * WIDTH + lane) % 257).unwrap_or_default();
                    f32::from(value) / 128.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn expert_bytes(
    model: &Path,
    content: &spm_gguf::Content,
    name: &str,
    expert: usize,
    bytes_per_expert: usize,
) -> Result<Vec<u8>, String> {
    let tensor = weights::tensor(content, name)?;
    let start = expert
        .checked_mul(bytes_per_expert)
        .ok_or("expert offset overflow")?;
    spm_gguf::read_tensor_range(
        model,
        tensor,
        start as u64,
        bytes_per_expert as u64,
        bytes_per_expert as u64,
    )
}

type Routes = (Vec<Vec<f32>>, Vec<Vec<usize>>);

fn route(model: &Path, content: &spm_gguf::Content, raw: &[Vec<f32>]) -> Result<Routes, String> {
    let scale = weights::load(model, content, "blk.0.ffn_gate_inp.scale")?;
    let router_inputs: Vec<_> = raw
        .iter()
        .map(|input| {
            let rms = (input.iter().map(|x| x * x).sum::<f32>() / WIDTH_F32 + 1e-6)
                .sqrt()
                .recip()
                / WIDTH_F32.sqrt();
            input
                .iter()
                .zip(&scale)
                .map(|(value, scale)| value * scale * rms)
                .collect()
        })
        .collect();
    let logits = weights::project_batch(
        model,
        content,
        "blk.0.ffn_gate_inp.weight",
        WIDTH,
        &router_inputs,
    )?;
    let probabilities: Vec<_> = logits.iter().map(|row| math::softmax(row)).collect();
    let routes = probabilities
        .iter()
        .map(|row| math::top_k(row, USED).iter().map(|item| item.0).collect())
        .collect();
    Ok((probabilities, routes))
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * value * (1.0 + 0.044_715 * value * value)).tanh())
}

fn direct_expert(
    model: &Path,
    content: &spm_gguf::Content,
    expert: usize,
    input: &[&[f32]],
) -> Result<Vec<Vec<f32>>, String> {
    let gate_up = expert_bytes(
        model,
        content,
        "blk.0.ffn_gate_up_exps.weight",
        expert,
        2 * FF * WIDTH / Q5_VALUES * Q5_BYTES,
    )?;
    let values = spm_gguf::decode_q5_k(&gate_up)?;
    let projected: Vec<_> = input
        .iter()
        .map(|row| math::matvec(&values, WIDTH, row))
        .collect::<Result<_, _>>()?;
    let active: Vec<Vec<f32>> = projected
        .iter()
        .map(|row| {
            row[..FF]
                .iter()
                .zip(&row[FF..])
                .map(|(gate, up)| gelu(*gate) * up)
                .collect()
        })
        .collect();
    let down = expert_bytes(
        model,
        content,
        "blk.0.ffn_down_exps.weight",
        expert,
        WIDTH * FF / Q8_VALUES * Q8_BYTES,
    )?;
    let values = spm_gguf::decode_q8_0(&down)?;
    let scale = weights::load(model, content, "blk.0.ffn_down_exps.scale")?[expert];
    active
        .iter()
        .map(|row| {
            Ok(math::matvec(&values, FF, row)?
                .into_iter()
                .map(|value| value * scale)
                .collect())
        })
        .collect()
}

fn descriptor(
    rows: usize,
    cols: usize,
    group_size: usize,
    encoding: Encoding,
) -> Result<OpDescriptor, String> {
    Ok(OpDescriptor {
        rows: u32::try_from(rows).map_err(|_| "row count exceeds wire format")?,
        cols: u32::try_from(cols).map_err(|_| "column count exceeds wire format")?,
        group_size: u32::try_from(group_size).map_err(|_| "group size exceeds wire format")?,
        encoding,
        lane_count: 1,
    })
}

fn emit(
    model: &Path,
    content: &spm_gguf::Content,
    output: &Path,
    selected: &[usize],
) -> Result<(), String> {
    let descriptors: Result<Vec<_>, _> = selected
        .iter()
        .flat_map(|_| {
            [
                descriptor(2 * FF, WIDTH, Q5_VALUES, Encoding::Q5K),
                descriptor(WIDTH, FF, Q8_VALUES, Encoding::Q8_0),
            ]
        })
        .collect();
    let mut writer = SpmWriter::new(descriptors?);
    for &expert in selected {
        let gate_up = expert_bytes(
            model,
            content,
            "blk.0.ffn_gate_up_exps.weight",
            expert,
            2 * FF * WIDTH / Q5_VALUES * Q5_BYTES,
        )?;
        for block in gate_up.chunks_exact(Q5_BYTES) {
            writer
                .write_raw_group(1.0, block, Q5_VALUES)
                .map_err(|e| e.to_string())?;
        }
        let down = expert_bytes(
            model,
            content,
            "blk.0.ffn_down_exps.weight",
            expert,
            WIDTH * FF / Q8_VALUES * Q8_BYTES,
        )?;
        for block in down.chunks_exact(Q8_BYTES) {
            writer
                .write_raw_group(1.0, block, Q8_VALUES)
                .map_err(|e| e.to_string())?;
        }
    }
    fs::write(output, writer.finish().map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn stream_matrix(
    groups: &mut GroupStream<FileWeightStream>,
    rows: usize,
    cols: usize,
    encoding: Encoding,
    inputs: &[&[f32]],
) -> Result<Vec<Vec<f32>>, String> {
    let block_values = if encoding == Encoding::Q5K {
        Q5_VALUES
    } else {
        Q8_VALUES
    };
    let mut output = vec![vec![0.0; rows]; inputs.len()];
    for at in (0..rows * cols).step_by(block_values) {
        let group = groups
            .next_group()
            .ok_or("missing Gemma expert group")?
            .map_err(|e| e.to_string())?;
        if group.encoding != encoding || group.count as usize != block_values {
            return Err("unexpected Gemma expert group".into());
        }
        let decoded = if encoding == Encoding::Q5K {
            spm_gguf::decode_q5_k(group.packed)?
        } else {
            spm_gguf::decode_q8_0(group.packed)?
        };
        for (offset, weight) in decoded.iter().enumerate() {
            let index = at + offset;
            let (row, col) = (index / cols, index % cols);
            for (result, input) in output.iter_mut().zip(inputs) {
                result[row] = weight.mul_add(input[col], result[row]);
            }
        }
    }
    Ok(output)
}

pub fn run(model: &Path, output: &Path, batch: usize) -> Result<(), String> {
    if batch == 0 {
        return Err("Gemma expert batch must be nonzero".into());
    }
    let content = spm_gguf::read(model)?;
    if content
        .metadata
        .get("general.architecture")
        .map(String::as_str)
        != Some("gemma4")
    {
        return Err("model is not Gemma-4".into());
    }
    let raw = inputs(batch);
    let (probabilities, routes) = route(model, &content, &raw)?;
    let norm = weights::load(model, &content, "blk.0.pre_ffw_norm_2.weight")?;
    let expert_inputs: Vec<_> = raw.iter().map(|row| math::rms_norm(row, &norm)).collect();
    let selected: Vec<_> = routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let started = Instant::now();
    emit(model, &content, output, &selected)?;
    let emit_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut groups = GroupStream::open(FileWeightStream::open(output).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut direct = vec![vec![0.0; WIDTH]; batch];
    let mut streamed = vec![vec![0.0; WIDTH]; batch];
    let mut direct_time = Duration::ZERO;
    let mut serial_time = Duration::ZERO;
    for &expert in &selected {
        let tokens: Vec<_> = routes
            .iter()
            .enumerate()
            .filter_map(|(token, ids)| ids.contains(&expert).then_some(token))
            .collect();
        let refs: Vec<_> = tokens
            .iter()
            .map(|&token| expert_inputs[token].as_slice())
            .collect();
        let started = Instant::now();
        let oracle = direct_expert(model, &content, expert, &refs)?;
        direct_time += started.elapsed();
        let started = Instant::now();
        let projected = stream_matrix(&mut groups, 2 * FF, WIDTH, Encoding::Q5K, &refs)?;
        let active: Vec<Vec<f32>> = projected
            .iter()
            .map(|row| {
                row[..FF]
                    .iter()
                    .zip(&row[FF..])
                    .map(|(gate, up)| gelu(*gate) * up)
                    .collect()
            })
            .collect();
        let active_refs: Vec<_> = active.iter().map(Vec::as_slice).collect();
        let mut serial = stream_matrix(&mut groups, WIDTH, FF, Encoding::Q8_0, &active_refs)?;
        let scale = weights::load(model, &content, "blk.0.ffn_down_exps.scale")?[expert];
        for row in &mut serial {
            row.iter_mut().for_each(|value| *value *= scale);
        }
        serial_time += started.elapsed();
        for (index, &token) in tokens.iter().enumerate() {
            let normalization: f32 = routes[token]
                .iter()
                .map(|&id| probabilities[token][id])
                .sum();
            let weight = probabilities[token][expert] / normalization;
            for lane in 0..WIDTH {
                direct[token][lane] += oracle[index][lane] * weight;
                streamed[token][lane] += serial[index][lane] * weight;
            }
        }
    }
    let direct_ms = direct_time.as_secs_f64() * 1000.0;
    let serial_ms = serial_time.as_secs_f64() * 1000.0;
    let max_error = direct
        .iter()
        .flatten()
        .zip(streamed.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if max_error > 0.002 {
        return Err(format!(
            "Gemma packed expert error {max_error} exceeds 0.002"
        ));
    }
    println!(
        "gemma_expert_smoke batch={batch} assignments={} selected_union={} bytes={} resident={} emit_ms={emit_ms:.3} direct_ms={direct_ms:.3} serial_ms={serial_ms:.3} max_error={max_error:.8}",
        batch * USED,
        selected.len(),
        fs::metadata(output).map_err(|e| e.to_string())?.len(),
        groups.resident_parameter_bytes()
    );
    Ok(())
}
