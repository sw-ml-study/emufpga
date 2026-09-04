use crate::{
    attention, layout,
    math::{add_scaled, rms_norm, top_k},
    moe::{self, Trace},
    weights,
};
use spm_gguf::Content;
use spm_viz_trace::{ExpertEvent, Trace as VizTrace};
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

const WIDTH: usize = 1024;
const LAYERS: usize = 24;

#[derive(Default)]
struct FullReport {
    bytes: u64,
    useful: u64,
    emit: Duration,
    read: Duration,
    decode: Duration,
    compute: Duration,
    peak_input: usize,
    expert_max: f32,
    combined_max: f32,
    trace: Option<VizTrace>,
}

macro_rules! record_full {
    ($full:expr, $report:expr, $emit:expr) => {{
        $full.bytes += $report.bytes;
        $full.useful += $report.useful;
        $full.emit += $emit;
        $full.read += $report.read;
        $full.decode += $report.decode;
        $full.compute += $report.compute;
        $full.peak_input = $full.peak_input.max($report.peak_input);
        $full.expert_max = $full.expert_max.max($report.expert_max);
        $full.combined_max = $full.combined_max.max($report.combined_max);
    }};
}

macro_rules! print_full {
    ($full:expr, $batch:expr) => {{
        let activation = $batch * (3 * WIDTH + 3 * 512) * 4;
        let kv = $batch * LAYERS * 8 * 64 * 2 * 4;
        println!(
            "spm_full layers={LAYERS} bytes={} useful={} transitions={} rewinds=0 peak_input={} activation_bound={} kv_f32={} emit_ms={:.3} read_ms={:.3} decode_ms={:.3} compute_ms={:.3} expert_max={:.8} combined_max={:.8}",
            $full.bytes, $full.useful, LAYERS - 1, $full.peak_input, activation, kv,
            $full.emit.as_secs_f64() * 1_000.0, $full.read.as_secs_f64() * 1_000.0,
            $full.decode.as_secs_f64() * 1_000.0, $full.compute.as_secs_f64() * 1_000.0,
            $full.expert_max, $full.combined_max
        );
    }};
}

macro_rules! record_trace {
    ($full:expr, $trace:expr, $report:expr, $layer:expr) => {
        if let Some(viz) = $full.trace.as_mut() {
            for expert in 0..32 {
                let routed = $trace
                    .routes
                    .iter()
                    .filter(|route| route.contains(&expert))
                    .count();
                viz.push(ExpertEvent {
                    layer: u8::try_from($layer).map_err(|_| "layer does not fit trace")?,
                    expert: u8::try_from(expert).map_err(|_| "expert does not fit trace")?,
                    selected: routed > 0,
                    routed_tokens: u16::try_from(routed).map_err(|_| "route count too large")?,
                    packed_bytes: 1_314_816,
                    decoded_bytes: if routed > 0 { 6_291_456 } else { 0 },
                    layer_read_us: u64::try_from($report.read.as_micros())
                        .map_err(|_| "read timing too large")?,
                    layer_decode_us: u64::try_from($report.decode.as_micros())
                        .map_err(|_| "decode timing too large")?,
                    layer_compute_us: u64::try_from($report.compute.as_micros())
                        .map_err(|_| "compute timing too large")?,
                })
                .map_err(str::to_owned)?;
            }
        }
    };
}

macro_rules! print_spm_report {
    ($report:expr, $emit:expr, $batch:expr, $schedule:expr) => {{
        let report = $report;
        let execution = report.read + report.decode + report.compute;
        let token_count = u32::try_from($batch).map_err(|_| "batch too large")?;
        let byte_count = u32::try_from(report.bytes).map_err(|_| "artifact too large")?;
        let tokens_per_second = f64::from(token_count) / execution.as_secs_f64();
        let read_gbps = f64::from(byte_count) / report.read.as_secs_f64() / 1_000_000_000.0;
        println!(
            "spm_q6 schedule={} batch={} selected={} bytes={} useful={} streams={} rewinds=0 resident={} peak_input={} emit_ms={:.3} warm_read_ms={:.3} decode_ms={:.3} compute_ms={:.3} read_gbps={read_gbps:.3} tokens_s={tokens_per_second:.3} cold_read=unmeasured expert_max={:.8} combined_max={:.8}",
            $schedule, $batch, report.selected_experts, report.bytes, report.useful,
            report.streams, report.resident, report.peak_input,
            $emit.as_secs_f64() * 1_000.0, report.read.as_secs_f64() * 1_000.0,
            report.decode.as_secs_f64() * 1_000.0, report.compute.as_secs_f64() * 1_000.0,
            report.expert_max, report.combined_max
        );
    }};
}

macro_rules! run_spm_layer {
    ($path:expr, $content:expr, $normalized:expr, $trace:expr, $layer:expr, $full:expr, $output:expr) => {{
        let output = env::var_os("SPM_GRANITE_FULL_SPM_PATH").or_else(|| {
            ($layer == 0)
                .then(|| env::var_os("SPM_GRANITE_SPM_PATH"))
                .flatten()
        });
        if let Some(output) = output {
            let (report, emit_time, streamed) = layout::verify(
                $path,
                $content,
                Path::new(&output),
                $normalized,
                $trace,
                $layer,
            )?;
            let schedule = if env::var_os("SPM_GRANITE_SELECTED_UNION").is_some() {
                "selected-union"
            } else {
                "all-expert"
            };
            println!("spm_layer={}", $layer);
            print_spm_report!(&report, emit_time, $normalized.len(), schedule);
            record_full!($full, &report, emit_time);
            record_trace!($full, $trace, &report, $layer);
            $output = streamed;
            if report.expert_max > 0.002 || report.combined_max > 0.002 {
                return Err("SPM execution differs from GGUF Rust oracle".into());
            }
        }
    }};
}

fn embeddings(path: &Path, content: &Content, tokens: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    let info = weights::tensor(content, "token_embd.weight")?;
    let row_bytes = (WIDTH / 256) * 210;
    tokens
        .iter()
        .map(|token| {
            let start = token
                .checked_mul(row_bytes)
                .ok_or("token offset overflow")? as u64;
            let bytes =
                spm_gguf::read_tensor_range(path, info, start, row_bytes as u64, row_bytes as u64)?;
            Ok(spm_gguf::decode_q6_k(&bytes)?
                .into_iter()
                .map(|value| value * 12.0)
                .collect())
        })
        .collect()
}

fn load_reference(path: &Path, name: &str) -> Result<Vec<f32>, String> {
    let file = path.join(format!("{name}.f32"));
    let bytes = fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn compare(
    reference: Option<&Path>,
    name: &str,
    actual: &[f32],
    tolerance: f32,
) -> Result<(), String> {
    let Some(directory) = reference else {
        return Ok(());
    };
    let expected = load_reference(directory, name)?;
    if expected.len() != actual.len() {
        return Err(format!("{name} length mismatch"));
    }
    let max = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let dot: f64 = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let norms = |values: &[f32]| {
        values
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let cosine = dot / (norms(&expected) * norms(actual));
    println!("compare_{name} max_abs={max:.8} cosine={cosine:.12}");
    if max > tolerance {
        return Err(format!("{name} exceeds tolerance {tolerance}"));
    }
    Ok(())
}

fn compare_trace(reference_dir: Option<&Path>, trace: &Trace) -> Result<(), String> {
    compare(
        reference_dir,
        "ffn_moe_logits-0",
        &trace.logits.concat(),
        0.1,
    )?;
    let Some(directory) = reference_dir else {
        return Ok(());
    };
    let raw = fs::read(directory.join("ffn_moe_topk-0.i32")).map_err(|error| error.to_string())?;
    let sorted: Vec<_> = raw
        .chunks_exact(4)
        .map(|chunk| {
            usize::try_from(i32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .map_err(|_| "negative expert ID")
        })
        .collect::<Result<_, _>>()?;
    let expected: Vec<_> = (0..trace.routes.len())
        .flat_map(|token| sorted[token * 32..token * 32 + 8].iter().copied())
        .collect();
    let actual = trace.routes.concat();
    if expected != actual {
        return Err(format!(
            "block 0 selected experts differ: oracle={expected:?} rust={actual:?}"
        ));
    }
    compare(
        Some(directory),
        "ffn_moe_weights_norm-0",
        &trace.weights.concat(),
        0.002,
    )?;
    let contributions: Vec<_> = trace
        .contributions
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect();
    compare(Some(directory), "ffn_moe_down-0", &contributions, 0.5)?;
    compare(
        Some(directory),
        "ffn_moe_out-0",
        &trace.output.concat(),
        0.5,
    )
}

fn compare_top(path: Option<PathBuf>, actual: &[(usize, f32)]) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let text = fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let expected: Vec<usize> = text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next()?.parse().ok())
        .collect();
    let overlap = actual
        .iter()
        .filter(|item| expected.contains(&item.0))
        .count();
    println!("compare_final top1={} overlap={overlap}/10", actual[0].0);
    if actual[0].0 != expected[0] || overlap < 9 {
        return Err("final ranking differs beyond tolerance".into());
    }
    Ok(())
}

fn block(
    path: &Path,
    content: &Content,
    hidden: &[Vec<f32>],
    layer: usize,
    reference: Option<&Path>,
    full: &mut FullReport,
) -> Result<Vec<Vec<f32>>, String> {
    let attn_norm = weights::load(path, content, &format!("blk.{layer}.attn_norm.weight"))?;
    let normalized: Vec<_> = hidden
        .iter()
        .map(|value| rms_norm(value, &attn_norm))
        .collect();
    let ffn_input = attention::run(path, content, hidden, &normalized, layer)?;
    if layer == 0 {
        compare(reference, "ffn_inp-0", &ffn_input.concat(), 2.0)?;
    }
    let ffn_norm = weights::load(path, content, &format!("blk.{layer}.ffn_norm.weight"))?;
    let normalized: Vec<_> = ffn_input
        .iter()
        .map(|value| rms_norm(value, &ffn_norm))
        .collect();
    if layer == 0 {
        compare(reference, "ffn_norm-0", &normalized.concat(), 2.0)?;
    }
    let trace = moe::run(path, content, &normalized, layer)?;
    let mut moe_output = trace.output.clone();
    if layer == 0 {
        compare_trace(reference, &trace)?;
    }
    run_spm_layer!(path, content, &normalized, &trace, layer, full, moe_output);
    Ok(ffn_input
        .iter()
        .zip(moe_output)
        .map(|(residual, value)| add_scaled(residual, &value, 0.22))
        .collect())
}

pub fn run(path: &Path, tokens: &[usize]) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("at least one token is required".into());
    }
    let content = spm_gguf::read(path)?;
    let reference_dir = env::var_os("SPM_GRANITE_REFERENCE_DIR").map(PathBuf::from);
    let mut hidden = embeddings(path, &content, tokens)?;
    let mut full = FullReport::default();
    if env::var_os("SPM_GRANITE_TRACE_JSON").is_some() {
        full.trace = Some(VizTrace {
            model: "granite-3.1-1b-a400m-q6_k",
            schedule: "all-expert",
            events: Vec::new(),
        });
    }
    compare(reference_dir.as_deref(), "inp_embd", &hidden.concat(), 0.1)?;
    for layer in 0..LAYERS {
        hidden = block(
            path,
            &content,
            &hidden,
            layer,
            reference_dir.as_deref(),
            &mut full,
        )?;
        if layer == 0 && env::var_os("SPM_GRANITE_BLOCK0_ONLY").is_some() {
            return Ok(());
        }
    }
    let norm = weights::load(path, &content, "output_norm.weight")?;
    let final_hidden = rms_norm(hidden.last().ok_or("missing final token")?, &norm);
    let mut logits =
        weights::project_stream(path, &content, "token_embd.weight", WIDTH, &final_hidden)?;
    logits.iter_mut().for_each(|value| *value /= 6.0);
    let top = top_k(&logits, 10);
    compare_top(
        env::var_os("SPM_GRANITE_LOGITS_GOLDEN").map(PathBuf::from),
        &top,
    )?;
    if env::var_os("SPM_GRANITE_FULL_SPM_PATH").is_some() {
        print_full!(&full, tokens.len());
    }
    if let (Some(path), Some(trace)) = (env::var_os("SPM_GRANITE_TRACE_JSON"), &full.trace) {
        fs::write(Path::new(&path), trace.to_json()).map_err(|error| error.to_string())?;
    }
    for (token, logit) in top {
        println!("{token} {logit:.9} {:08x}", logit.to_bits());
    }
    Ok(())
}
