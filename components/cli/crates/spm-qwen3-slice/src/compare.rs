use std::{fs, path::Path};

pub struct Metrics {
    pub max_abs: f32,
    pub mean_abs: f64,
    pub cosine: f64,
}

fn read_vector(path: &Path, len: usize, position: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let needed = len.checked_mul(4).ok_or("reference length overflow")?;
    let start = position
        .checked_mul(needed)
        .ok_or("reference offset overflow")?;
    let end = start.checked_add(needed).ok_or("reference end overflow")?;
    if bytes.len() < end {
        return Err(format!("{} lacks vector {position}", path.display()));
    }
    Ok(bytes[start..end]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

pub fn measure(
    directory: &Path,
    name: &str,
    position: usize,
    actual: &[f32],
) -> Result<Metrics, String> {
    let expected = read_vector(
        &directory.join(format!("{name}.f32")),
        actual.len(),
        position,
    )?;
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut expected_sq = 0.0_f64;
    let mut actual_sq = 0.0_f64;
    for (&left, &right) in expected.iter().zip(actual) {
        let difference = (left - right).abs();
        max_abs = max_abs.max(difference);
        sum_abs += f64::from(difference);
        dot += f64::from(left) * f64::from(right);
        expected_sq += f64::from(left) * f64::from(left);
        actual_sq += f64::from(right) * f64::from(right);
    }
    let mean_abs = sum_abs / f64::from(u16::try_from(actual.len()).map_err(|_| "vector too long")?);
    let cosine = dot / (expected_sq.sqrt() * actual_sq.sqrt());
    Ok(Metrics {
        max_abs,
        mean_abs,
        cosine,
    })
}

pub fn check(
    directory: &Path,
    name: &str,
    position: usize,
    actual: &[f32],
    max_abs: f32,
    min_cosine: f64,
) -> Result<(), String> {
    let metrics = measure(directory, name, position, actual)?;
    println!(
        "compare_{name}[{position}] max_abs={:.9} mean_abs={:.9} cosine={:.12}",
        metrics.max_abs, metrics.mean_abs, metrics.cosine
    );
    if metrics.max_abs > max_abs || metrics.cosine < min_cosine {
        return Err(format!("{name} exceeds reference tolerance"));
    }
    Ok(())
}
