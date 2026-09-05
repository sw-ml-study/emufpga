use crate::{
    math::{softmax, top_k},
    moe::Trace,
    shape::MoeShape,
    weights,
};
use spm_codec_dense::encode_into;
use spm_file::SpmWriter;
use spm_layout::{Encoding, OpDescriptor};
use spm_linear::streamed;
use spm_stream::WeightStream;
use spm_stream_file::{DEFAULT_CAPACITY, FileWeightStream, PrefetchFileWeightStream};
use spm_stream_groups::GroupStream;
use std::{
    collections::BTreeSet,
    env, fs,
    path::Path,
    time::{Duration, Instant},
};

const Q6_VALUES: usize = 256;
const Q6_BYTES: usize = 210;

pub struct Report {
    pub bytes: u64,
    pub useful: u64,
    pub streams: usize,
    pub resident: usize,
    pub peak_input: usize,
    pub selected_experts: usize,
    pub read: Duration,
    pub decode: Duration,
    pub compute: Duration,
    pub expert_max: f32,
    pub combined_max: f32,
}

#[derive(Default)]
struct Timing {
    read: Duration,
    decode: Duration,
    compute: Duration,
}

#[derive(Clone, Copy)]
struct ExpertContext<'a> {
    inputs: &'a [Vec<f32>],
    probabilities: &'a [Vec<f32>],
    trace: &'a Trace,
    shape: MoeShape,
}

macro_rules! descriptor {
    ($rows:expr, $cols:expr, $encoding:expr) => {
        OpDescriptor {
            rows: u32::try_from($rows).expect("rows fit u32"),
            cols: u32::try_from($cols).expect("columns fit u32"),
            group_size: if $encoding == Encoding::Q6K {
                u32::try_from(Q6_VALUES).expect("Q6 group fits u32")
            } else {
                u32::try_from($rows).expect("rows fit u32")
            },
            encoding: $encoding,
            lane_count: 1,
        }
    };
}

macro_rules! append_f32 {
    ($writer:expr, $matrix:expr, $rows:expr, $cols:expr) => {
        for column in 0..$cols {
            let values: Vec<_> = (0..$rows)
                .map(|row| $matrix[row * $cols + column])
                .collect();
            let mut bytes = vec![0; values.len() * 4];
            encode_into(&values, &mut bytes).map_err(|needed| format!("need {needed} bytes"))?;
            $writer
                .write_raw_group(1.0, &bytes, $rows)
                .map_err(|error| error.to_string())?;
        }
    };
}

macro_rules! activate {
    ($gate:expr, $up:expr) => {
        $gate
            .iter()
            .zip($up)
            .map(|(gate, up)| {
                gate.iter()
                    .zip(up)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>()
    };
}

macro_rules! validate_routes {
    ($probabilities:expr, $trace:expr, $layer:expr, $shape:expr) => {
        for (token, values) in $probabilities.iter().enumerate() {
            let selected: Vec<_> = top_k(values, $shape.used)
                .iter()
                .map(|item| item.0)
                .collect();
            if selected != $trace.routes[token] {
                return Err(format!(
                    "layer {} token {token} packed router differs",
                    $layer
                ));
            }
        }
    };
}

macro_rules! combined_error {
    ($combined:expr, $trace:expr) => {
        $combined
            .iter()
            .flatten()
            .zip($trace.output.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    };
}

macro_rules! reuse_artifact {
    ($output:expr) => {
        if env::var_os("SPM_GRANITE_REUSE_SPM").is_some() {
            fs::metadata($output)
                .map_err(|error| format!("reuse {}: {error}", $output.display()))?;
            return Ok(Duration::ZERO);
        }
    };
}

fn emit(
    model: &Path,
    content: &spm_gguf::Content,
    output: &Path,
    shape: MoeShape,
    experts: &[usize],
    layer: usize,
) -> Result<Duration, String> {
    reuse_artifact!(output);
    let started = Instant::now();
    let mut shapes = vec![(shape.experts, shape.width, Encoding::F32)];
    shapes.extend(experts.iter().flat_map(|_| {
        [
            (shape.ff, shape.width, Encoding::Q6K),
            (shape.ff, shape.width, Encoding::Q6K),
            (shape.width, shape.ff, Encoding::Q6K),
        ]
    }));
    let descriptors = shapes
        .iter()
        .map(|&(rows, cols, encoding)| descriptor!(rows, cols, encoding))
        .collect();
    let mut writer = SpmWriter::new(descriptors);
    let router = weights::load(model, content, &format!("blk.{layer}.ffn_gate_inp.weight"))?;
    append_f32!(&mut writer, &router, shape.experts, shape.width);
    for &expert in experts {
        for (name, cols, rows) in [
            ("ffn_up_exps", shape.width, shape.ff),
            ("ffn_gate_exps", shape.width, shape.ff),
            ("ffn_down_exps", shape.ff, shape.width),
        ] {
            let bytes = weights::expert_bytes(
                model,
                content,
                &format!("blk.{layer}.{name}.weight"),
                cols,
                rows,
                expert,
            )?;
            for block in bytes.chunks_exact(Q6_BYTES) {
                writer
                    .write_raw_group(1.0, block, Q6_VALUES)
                    .map_err(|error| error.to_string())?;
            }
        }
    }
    fs::write(output, writer.finish().map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    Ok(started.elapsed())
}

fn router(
    groups: &mut GroupStream<Box<dyn WeightStream>>,
    inputs: &[Vec<f32>],
    shape: MoeShape,
) -> Result<Vec<Vec<f32>>, String> {
    let flat: Vec<_> = inputs.iter().flatten().copied().collect();
    let mut output = vec![0.0; inputs.len() * shape.experts];
    streamed(
        groups,
        (shape.experts, shape.width),
        (&flat, inputs.len()),
        &mut output,
    )
    .map_err(|error| error.to_string())?;
    Ok(output
        .chunks_exact(shape.experts)
        .map(<[f32]>::to_vec)
        .collect())
}

fn q6_operation(
    groups: &mut GroupStream<Box<dyn WeightStream>>,
    shape: (usize, usize),
    inputs: &[&[f32]],
    timing: &mut Timing,
) -> Result<Vec<Vec<f32>>, String> {
    let mut outputs = vec![vec![0.0; shape.0]; inputs.len()];
    for at in (0..shape.0 * shape.1).step_by(Q6_VALUES) {
        let started = Instant::now();
        let group = groups
            .next_group()
            .ok_or("missing Q6_K group")?
            .map_err(|error| error.to_string())?;
        timing.read += started.elapsed();
        if inputs.is_empty() {
            continue;
        }
        if group.encoding != Encoding::Q6K || group.count as usize != Q6_VALUES {
            return Err("unexpected packed expert group".into());
        }
        let started = Instant::now();
        let decoded = spm_gguf::decode_q6_k(group.packed)?;
        timing.decode += started.elapsed();
        let started = Instant::now();
        for (weight_offset, weight) in decoded.iter().enumerate() {
            let index = at + weight_offset;
            let (row, col) = (index / shape.1, index % shape.1);
            for (output, input) in outputs.iter_mut().zip(inputs) {
                output[row] = weight.mul_add(input[col], output[row]);
            }
        }
        timing.compute += started.elapsed();
    }
    Ok(outputs)
}

fn process_expert(
    groups: &mut GroupStream<Box<dyn WeightStream>>,
    context: ExpertContext<'_>,
    expert: usize,
    combined: &mut [Vec<f32>],
    timing: &mut Timing,
) -> Result<f32, String> {
    let mut expert_max = 0.0_f32;
    let tokens: Vec<_> = context
        .trace
        .routes
        .iter()
        .enumerate()
        .filter_map(|(token, route)| route.contains(&expert).then_some(token))
        .collect();
    let selected: Vec<_> = tokens
        .iter()
        .map(|&token| context.inputs[token].as_slice())
        .collect();
    let up = q6_operation(
        groups,
        (context.shape.ff, context.shape.width),
        &selected,
        timing,
    )?;
    let gate = q6_operation(
        groups,
        (context.shape.ff, context.shape.width),
        &selected,
        timing,
    )?;
    let active = activate!(gate, up);
    let active_refs: Vec<_> = active.iter().map(Vec::as_slice).collect();
    let down = q6_operation(
        groups,
        (context.shape.width, context.shape.ff),
        &active_refs,
        timing,
    )?;
    for (&token, value) in tokens.iter().zip(down) {
        let slot = context.trace.routes[token]
            .iter()
            .position(|&id| id == expert)
            .ok_or("missing expert slot")?;
        expert_max = expert_max.max(
            value
                .iter()
                .zip(&context.trace.contributions[token][slot])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max),
        );
        let normalization: f32 = context.trace.routes[token]
            .iter()
            .map(|&id| context.probabilities[token][id])
            .sum();
        for (total, lane) in combined[token].iter_mut().zip(value) {
            *total += lane * context.probabilities[token][expert] / normalization;
        }
    }
    Ok(expert_max)
}

fn execute_experts(
    groups: &mut GroupStream<Box<dyn WeightStream>>,
    inputs: &[Vec<f32>],
    probabilities: &[Vec<f32>],
    trace: &Trace,
    experts: &[usize],
    shape: MoeShape,
    timing: &mut Timing,
) -> Result<(f32, Vec<Vec<f32>>), String> {
    let mut combined = vec![vec![0.0; shape.width]; inputs.len()];
    let mut expert_max = 0.0_f32;
    let context = ExpertContext {
        inputs,
        probabilities,
        trace,
        shape,
    };
    for &expert in experts {
        expert_max = expert_max.max(process_expert(
            groups,
            context,
            expert,
            &mut combined,
            timing,
        )?);
    }
    Ok((expert_max, combined))
}

fn report(
    output: &Path,
    groups: &GroupStream<Box<dyn WeightStream>>,
    selected: usize,
    expert_count: usize,
    shape: MoeShape,
    timing: &Timing,
    errors: (f32, f32),
) -> Result<Report, String> {
    let expert_stream_bytes = 3 * (shape.ff * shape.width / Q6_VALUES) * (Q6_BYTES + 4);
    Ok(Report {
        bytes: fs::metadata(output)
            .map_err(|error| error.to_string())?
            .len(),
        useful: (shape.width * (shape.experts * 4 + 4) + selected * expert_stream_bytes) as u64,
        streams: 1 + expert_count * 3,
        resident: groups.resident_parameter_bytes(),
        peak_input: 2 * DEFAULT_CAPACITY + groups.resident_parameter_bytes() + 4 * Q6_VALUES,
        selected_experts: selected,
        read: timing.read,
        decode: timing.decode,
        compute: timing.compute,
        expert_max: errors.0,
        combined_max: errors.1,
    })
}

pub fn verify(
    model: &Path,
    content: &spm_gguf::Content,
    output: &Path,
    inputs: &[Vec<f32>],
    trace: &Trace,
    layer: usize,
    shape: MoeShape,
) -> Result<(Report, Duration, Vec<Vec<f32>>), String> {
    shape.validate()?;
    let selected_set = trace
        .routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let experts: Vec<_> = if env::var_os("SPM_GRANITE_SELECTED_UNION").is_some() {
        selected_set.iter().copied().collect()
    } else {
        (0..shape.experts).collect()
    };
    let emit_ns = emit(model, content, output, shape, &experts, layer)?;
    let backend: Box<dyn WeightStream> = if env::var_os("SPM_GRANITE_PREFETCH").is_some() {
        Box::new(PrefetchFileWeightStream::open(output).map_err(|error| error.to_string())?)
    } else {
        Box::new(FileWeightStream::open(output).map_err(|error| error.to_string())?)
    };
    let mut groups = GroupStream::open(backend).map_err(|error| error.to_string())?;
    let logits = router(&mut groups, inputs, shape)?;
    let probabilities: Vec<_> = logits.iter().map(|values| softmax(values)).collect();
    validate_routes!(probabilities, trace, layer, shape);
    let mut timing = Timing::default();
    let (expert_max, combined) = execute_experts(
        &mut groups,
        inputs,
        &probabilities,
        trace,
        &experts,
        shape,
        &mut timing,
    )?;
    let combined_max = combined_error!(combined, trace);
    let report = report(
        output,
        &groups,
        selected_set.len(),
        experts.len(),
        shape,
        &timing,
        (expert_max, combined_max),
    )?;
    Ok((report, emit_ns, combined))
}
