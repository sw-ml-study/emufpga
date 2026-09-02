pub fn rms_norm(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let count = f32::from(u16::try_from(input.len()).expect("model width fits u16"));
    let mean_square = input.iter().map(|x| x * x).sum::<f32>() / count;
    let scale = (mean_square + 1e-6).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * scale * w)
        .collect()
}

pub fn matvec(weights: &[f32], cols: usize, input: &[f32]) -> Result<Vec<f32>, String> {
    if input.len() != cols || weights.len() % cols != 0 {
        return Err("invalid matrix/vector shape".into());
    }
    Ok(weights
        .chunks_exact(cols)
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect())
}

pub fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

pub fn print_stats(label: &str, values: &[f32]) {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let count = f32::from(u16::try_from(values.len()).expect("model width fits u16"));
    let rms = (values.iter().map(|x| x * x).sum::<f32>() / count).sqrt();
    println!(
        "{label} len={} min={min:.8} max={max:.8} rms={rms:.8} first={:.8}",
        values.len(),
        values[0]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_known_vector() {
        let result = rms_norm(&[3.0, 4.0], &[1.0, 2.0]);
        assert!((result[0] - 0.848_528).abs() < 1e-5);
        assert!((result[1] - 2.262_741_6).abs() < 1e-5);
    }

    #[test]
    fn matvec_known_matrix() {
        assert_eq!(
            matvec(&[1.0, 2.0, 3.0, 4.0], 2, &[5.0, 6.0]).unwrap(),
            [17.0, 39.0]
        );
    }
}
