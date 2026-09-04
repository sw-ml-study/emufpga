use crate::{math::softmax, moe::Trace, weights};
use spm_codec_dense::encode_into;
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_linear::streamed;
use spm_stream_file::FileWeightStream;
use spm_stream_groups::GroupStream;
use std::{collections::BTreeSet, fs, path::Path};

const WIDTH: usize = 1024;
const FF: usize = 512;

pub struct Report {
    pub bytes: u64,
    pub streams: usize,
    pub resident: usize,
    pub expert_max: f32,
    pub combined_max: f32,
}

fn descriptor(rows: usize, cols: usize) -> OpDescriptor {
    OpDescriptor {
        rows: u32::try_from(rows).expect("rows fit u32"),
        cols: u32::try_from(cols).expect("columns fit u32"),
        group_size: u32::try_from(rows).expect("rows fit u32"),
        encoding: Encoding::F32,
        lane_count: 1,
    }
}

fn append(writer: &mut SpmWriter, matrix: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    for column in 0..cols {
        let values: Vec<_> = (0..rows).map(|row| matrix[row * cols + column]).collect();
        let mut bytes = vec![0; values.len() * 4];
        encode_into(&values, &mut bytes).map_err(|needed| format!("need {needed} bytes"))?;
        writer
            .write_raw_group(1.0, &bytes, rows)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn matrices(
    path: &Path,
    content: &spm_gguf::Content,
    experts: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let mut result = vec![weights::load(path, content, "blk.0.ffn_gate_inp.weight")?];
    for &expert in experts {
        for (name, cols, rows) in [
            ("ffn_up_exps", WIDTH, FF),
            ("ffn_gate_exps", WIDTH, FF),
            ("ffn_down_exps", FF, WIDTH),
        ] {
            result.push(weights::expert_matrix(
                path,
                content,
                &format!("blk.0.{name}.weight"),
                cols,
                rows,
                expert,
            )?);
        }
    }
    Ok(result)
}

fn emit(path: &Path, matrices: &[Vec<f32>], experts: usize) -> Result<(), String> {
    let mut shapes = vec![(32, WIDTH)];
    shapes.extend((0..experts).flat_map(|_| [(FF, WIDTH), (FF, WIDTH), (WIDTH, FF)]));
    let descriptors: Vec<_> = shapes
        .iter()
        .map(|&(rows, cols)| descriptor(rows, cols))
        .collect();
    let mut writer = SpmWriter::new(descriptors);
    for (matrix, &(rows, cols)) in matrices.iter().zip(&shapes) {
        append(&mut writer, matrix, rows, cols)?;
    }
    fs::write(path, writer.finish().map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())
}

fn projection(
    groups: &mut GroupStream<FileWeightStream>,
    shape: (usize, usize),
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let mut output = vec![0.0; shape.0];
    streamed(groups, shape, (input, 1), &mut output).map_err(|error| error.to_string())?;
    Ok(output)
}

fn execute_experts(
    groups: &mut GroupStream<FileWeightStream>,
    experts: &[usize],
    input: &[f32],
    probabilities: &[f32],
    trace: &Trace,
) -> Result<(f32, Vec<f32>), String> {
    let sum: f32 = trace.routes[0].iter().map(|&id| probabilities[id]).sum();
    let mut combined = vec![0.0; WIDTH];
    let mut expert_max = 0.0_f32;
    for &expert in experts {
        let up = projection(groups, (FF, WIDTH), input)?;
        let gate = projection(groups, (FF, WIDTH), input)?;
        let active: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = projection(groups, (WIDTH, FF), &active)?;
        let slot = trace.routes[0]
            .iter()
            .position(|&id| id == expert)
            .ok_or("expert slot missing")?;
        let differences = down
            .iter()
            .zip(&trace.contributions[0][slot])
            .map(|(a, b)| (a - b).abs());
        expert_max = expert_max.max(differences.fold(0.0, f32::max));
        for (total, value) in combined.iter_mut().zip(down) {
            *total += value * probabilities[expert] / sum;
        }
    }
    Ok((expert_max, combined))
}

pub fn verify(
    model: &Path,
    content: &spm_gguf::Content,
    output: &Path,
    input: &[f32],
    trace: &Trace,
) -> Result<Report, String> {
    let experts: Vec<_> = trace.routes[0]
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let all = matrices(model, content, &experts)?;
    emit(output, &all, experts.len())?;
    drop(all);
    let mut groups =
        GroupStream::open(FileWeightStream::open(output).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let router = projection(&mut groups, (32, WIDTH), input)?;
    let probabilities = softmax(&router);
    let (expert_max, combined) =
        execute_experts(&mut groups, &experts, input, &probabilities, trace)?;
    Ok(Report {
        bytes: fs::metadata(output)
            .map_err(|error| error.to_string())?
            .len(),
        streams: 25,
        resident: groups.resident_parameter_bytes(),
        expert_max,
        combined_max: combined
            .iter()
            .zip(&trace.output[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max),
    })
}
