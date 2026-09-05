use crate::{layout, moe, shape};
use std::path::Path;

fn deterministic_inputs(batch: usize, width: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|token| {
            (0..width)
                .map(|lane| {
                    let value = (token * width + lane) % 257;
                    (f32::from(u16::try_from(value).expect("value is below 257")) - 128.0) / 128.0
                })
                .collect()
        })
        .collect()
}

pub fn run(model: &Path, output: &Path, batch: usize) -> Result<(), String> {
    if batch == 0 {
        return Err("expert smoke batch must be nonzero".into());
    }
    let content = spm_gguf::read(model)?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .ok_or("GGUF has no general.architecture")?;
    let shape = shape::for_architecture(architecture)
        .ok_or_else(|| format!("unsupported MoE architecture {architecture}"))?;
    shape.validate()?;
    let inputs = deterministic_inputs(batch, shape.width);
    let trace = moe::run(model, &content, &inputs, 0, shape)?;
    let (report, emit, streamed) =
        layout::verify(model, &content, output, &inputs, &trace, 0, shape)?;
    let max_error = streamed
        .iter()
        .flatten()
        .zip(trace.output.iter().flatten())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if max_error > 0.002 {
        return Err(format!(
            "packed expert smoke error {max_error} exceeds 0.002"
        ));
    }
    println!(
        "expert_smoke architecture={architecture} batch={batch} width={} ff={} experts={} used={} selected_union={} bytes={} resident={} peak_input={} emit_ms={:.3} read_ms={:.3} decode_ms={:.3} compute_ms={:.3} max_error={max_error:.8}",
        shape.width,
        shape.ff,
        shape.experts,
        shape.used,
        report.selected_experts,
        report.bytes,
        report.resident,
        report.peak_input,
        emit.as_secs_f64() * 1000.0,
        report.read.as_secs_f64() * 1000.0,
        report.decode.as_secs_f64() * 1000.0,
        report.compute.as_secs_f64() * 1000.0,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_inputs_are_bounded_and_token_specific() {
        let values = deterministic_inputs(2, 256);
        assert_eq!(values.len(), 2);
        assert!(
            values
                .iter()
                .flatten()
                .all(|value| (-1.0..=1.0).contains(value))
        );
        assert_ne!(values[0], values[1]);
    }
}
