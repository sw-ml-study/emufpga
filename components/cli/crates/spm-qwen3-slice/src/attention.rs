const HEAD: usize = 128;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;

pub struct KvCache {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

pub fn rope(head: &mut [f32], position: usize) {
    let half = head.len() / 2;
    for index in 0..half {
        let index_f = f32::from(u16::try_from(index).expect("head index fits u16"));
        let width = f32::from(u16::try_from(head.len()).expect("head width fits u16"));
        let position = f32::from(u16::try_from(position).expect("position fits u16"));
        let exponent = 2.0 * index_f / width;
        let angle = position / 1_000_000_f32.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let left = head[index];
        let right = head[index + half];
        head[index] = left * cos - right * sin;
        head[index + half] = right * cos + left * sin;
    }
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub fn causal_gqa(queries: &[Vec<f32>], cache: &KvCache) -> Vec<Vec<f32>> {
    let mut output = Vec::with_capacity(queries.len());
    for (position, query) in queries.iter().enumerate() {
        let mut token = Vec::with_capacity(Q_HEADS * HEAD);
        for q_head in 0..Q_HEADS {
            let kv_head = q_head / (Q_HEADS / KV_HEADS);
            let query = &query[q_head * HEAD..(q_head + 1) * HEAD];
            let mut scores: Vec<_> = cache.keys[..=position]
                .iter()
                .map(|key| {
                    query
                        .iter()
                        .zip(&key[kv_head * HEAD..(kv_head + 1) * HEAD])
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        / 11.313_708
                })
                .collect();
            softmax(&mut scores);
            for lane in 0..HEAD {
                let value = cache.values[..=position]
                    .iter()
                    .zip(&scores)
                    .map(|(value, score)| value[kv_head * HEAD + lane] * score)
                    .sum();
                token.push(value);
            }
        }
        output.push(token);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_identity_and_known_pair() {
        let mut head: Vec<_> = (0_u16..128).map(f32::from).collect();
        let original = head.clone();
        rope(&mut head, 0);
        assert_eq!(head, original);
        let mut head = vec![0.0; HEAD];
        head[0] = 1.0;
        rope(&mut head, 1);
        assert!((head[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((head[HEAD / 2] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut values = [1000.0, 1000.0];
        softmax(&mut values);
        assert_eq!(values.map(f32::to_bits), [0.5_f32; 2].map(f32::to_bits));
    }

    #[test]
    fn causal_gqa_maps_four_query_heads_to_each_kv_head() {
        let queries = vec![vec![0.0; Q_HEADS * HEAD]];
        let keys = vec![vec![0.0; KV_HEADS * HEAD]];
        let values = vec![
            (0..KV_HEADS)
                .flat_map(|head| {
                    let value = f32::from(u16::try_from(head).expect("head fits u16"));
                    std::iter::repeat_n(value, HEAD)
                })
                .collect(),
        ];
        let output = causal_gqa(&queries, &KvCache { keys, values });
        for q_head in 0..Q_HEADS {
            let expected = f32::from(u16::try_from(q_head / 4).expect("head fits u16"));
            assert!((output[0][q_head * HEAD] - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn later_key_cannot_affect_earlier_output() {
        let queries = vec![vec![0.0; Q_HEADS * HEAD]; 2];
        let keys = vec![vec![0.0; KV_HEADS * HEAD]; 2];
        let mut values = vec![vec![1.0; KV_HEADS * HEAD]; 2];
        values[1].fill(99.0);
        let cache = KvCache { keys, values };
        assert_eq!(causal_gqa(&queries, &cache)[0], vec![1.0; Q_HEADS * HEAD]);
    }
}
