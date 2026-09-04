pub fn rms_norm(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let count = f32::from(u16::try_from(input.len()).expect("model width fits u16"));
    let scale = (input.iter().map(|x| x * x).sum::<f32>() / count + 1e-6)
        .sqrt()
        .recip();
    input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * scale * w)
        .collect()
}

pub fn matvec(weights: &[f32], cols: usize, input: &[f32]) -> Result<Vec<f32>, String> {
    if input.len() != cols || !weights.len().is_multiple_of(cols) {
        return Err("invalid matrix/vector shape".into());
    }
    Ok(weights
        .chunks_exact(cols)
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect())
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<_> = values.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = result.iter().sum();
    result.iter_mut().for_each(|value| *value /= sum);
    result
}

pub fn add_scaled(left: &[f32], right: &[f32], scale: f32) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b * scale).collect()
}

pub fn top_k(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<_> = values.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(count);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_is_stable() {
        assert_eq!(
            top_k(&[1.0, 3.0, 3.0, 2.0], 3),
            [(1, 3.0), (2, 3.0), (3, 2.0)]
        );
        let probabilities = softmax(&[1000.0, 1000.0]);
        assert_eq!(probabilities, [0.5, 0.5]);
    }
}
