use std::{fs, path::Path};

pub struct Metrics {
    pub max_abs: f32,
    pub mean_abs: f64,
    pub cosine: f64,
}

pub fn measure(
    directory: &Path,
    name: &str,
    position: usize,
    actual: &[f32],
) -> Result<Metrics, String> {
    let path = directory.join(format!("{name}.f32"));
    let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let needed = actual
        .len()
        .checked_mul(4)
        .ok_or("reference length overflow")?;
    let start = position
        .checked_mul(needed)
        .ok_or("reference offset overflow")?;
    let end = start.checked_add(needed).ok_or("reference end overflow")?;
    let slice = bytes
        .get(start..end)
        .ok_or_else(|| format!("{} lacks vector {position}", path.display()))?;
    let expected: Vec<_> = slice
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
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

pub fn check_stage(
    directory: Option<&Path>,
    name: &str,
    values: &[Vec<f32>],
    tolerances: (f32, f64),
) -> Result<(), String> {
    for (position, value) in values.iter().enumerate() {
        crate::math::print_stats(&format!("{name}[{position}]"), value);
        if let Some(directory) = directory {
            let metrics = measure(directory, name, position, value)?;
            println!(
                "compare_{name}[{position}] max_abs={:.9} mean_abs={:.9} cosine={:.12}",
                metrics.max_abs, metrics.mean_abs, metrics.cosine
            );
            if metrics.max_abs > tolerances.0 || metrics.cosine < tolerances.1 {
                return Err(format!("{name} exceeds reference tolerance"));
            }
        }
    }
    Ok(())
}

fn parse_top_line(line: &str) -> Result<(usize, f32), String> {
    let mut fields = line.split_whitespace();
    let token = fields.next().ok_or("golden token missing")?;
    let logit = fields.next().ok_or("golden logit missing")?;
    Ok((
        token.parse().map_err(|_| "invalid golden token")?,
        logit.parse().map_err(|_| "invalid golden logit")?,
    ))
}

pub fn check_top(path: Option<&Path>, actual: &[(usize, f32)], max_abs: f32) -> Result<(), String> {
    for &(token, logit) in actual {
        println!("{token} {logit:.9} {:08x}", logit.to_bits());
    }
    let Some(path) = path else { return Ok(()) };
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let expected = text
        .lines()
        .skip(1)
        .map(parse_top_line)
        .collect::<Result<Vec<_>, _>>()?;
    if actual.first().map(|item| item.0) != expected.first().map(|item| item.0) {
        return Err("top token differs from CPU oracle".into());
    }
    for &(token, logit) in actual {
        let reference = expected
            .iter()
            .find(|item| item.0 == token)
            .ok_or("top-ten set differs from CPU oracle")?;
        if (logit - reference.1).abs() > max_abs {
            return Err(format!("token {token} exceeds logits tolerance"));
        }
    }
    Ok(())
}
