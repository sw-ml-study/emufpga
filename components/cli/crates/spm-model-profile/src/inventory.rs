use serde::Serialize;
use spm_gguf::{Content, TensorInfo};
use std::collections::BTreeSet;

#[derive(Debug, Serialize)]
pub struct ExpertTensor {
    pub name: String,
    pub layer: usize,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub bytes: u64,
    pub bytes_per_expert: u64,
}

#[derive(Debug, Serialize)]
pub struct Inventory {
    pub layers: usize,
    pub experts_per_layer: u64,
    pub active_experts_per_token: Option<u64>,
    pub expert_tensor_bytes: u64,
    pub tensors: Vec<ExpertTensor>,
}

fn layer(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn is_expert(tensor: &TensorInfo) -> bool {
    tensor.name.ends_with("_exps.weight")
}

fn metadata_u64(content: &Content, suffix: &str) -> Option<u64> {
    content
        .metadata
        .iter()
        .find(|(key, _)| key.ends_with(suffix))
        .and_then(|(_, value)| value.parse().ok())
}

pub fn build(content: &Content) -> Result<Inventory, String> {
    let selected: Vec<_> = content
        .tensors
        .iter()
        .filter(|tensor| is_expert(tensor))
        .collect();
    let experts: BTreeSet<_> = selected
        .iter()
        .filter_map(|tensor| tensor.dims.last())
        .copied()
        .collect();
    if selected.is_empty() || experts.len() != 1 || experts.contains(&0) {
        return Err("missing or inconsistent routed-expert tensors".into());
    }
    let count = *experts.first().ok_or("missing expert count")?;
    let mut layers = BTreeSet::new();
    let mut tensors = Vec::new();
    for tensor in selected {
        let layer = layer(&tensor.name)
            .ok_or_else(|| format!("expert tensor has no layer: {}", tensor.name))?;
        if tensor.len % count != 0 {
            return Err(format!(
                "expert tensor is not divisible by expert count: {}",
                tensor.name
            ));
        }
        layers.insert(layer);
        tensors.push(ExpertTensor {
            name: tensor.name.clone(),
            layer,
            dimensions: tensor.dims.clone(),
            ggml_type: tensor.dtype,
            offset: tensor.offset,
            bytes: tensor.len,
            bytes_per_expert: tensor.len / count,
        });
    }
    let expert_tensor_bytes = tensors.iter().map(|tensor| tensor.bytes).sum();
    Ok(Inventory {
        layers: layers.len(),
        experts_per_layer: count,
        active_experts_per_token: metadata_u64(content, "expert_used_count"),
        expert_tensor_bytes,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn content(tensors: Vec<TensorInfo>) -> Content {
        Content {
            version: 3,
            metadata_count: 0,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors,
        }
    }

    #[test]
    fn inventories_expert_ranges_without_tensor_reads() {
        let tensor = TensorInfo {
            name: "blk.3.ffn_gate_up_exps.weight".into(),
            dims: vec![4, 8],
            dtype: 18,
            offset: 4096,
            len: 800,
        };
        let inventory = build(&content(vec![tensor])).unwrap();
        assert_eq!((inventory.layers, inventory.experts_per_layer), (1, 8));
        assert_eq!(
            (
                inventory.tensors[0].offset,
                inventory.tensors[0].bytes_per_expert
            ),
            (4096, 100)
        );
    }

    #[test]
    fn rejects_inconsistent_expert_shapes() {
        let tensor = |name: &str, experts| TensorInfo {
            name: name.into(),
            dims: vec![4, experts],
            dtype: 18,
            offset: 0,
            len: 800,
        };
        let result = build(&content(vec![
            tensor("blk.0.ffn_gate_up_exps.weight", 8),
            tensor("blk.1.ffn_down_exps.weight", 4),
        ]));
        assert!(result.unwrap_err().contains("inconsistent"));
    }
}
