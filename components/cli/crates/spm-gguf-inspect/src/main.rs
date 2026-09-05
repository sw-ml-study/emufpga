use spm_gguf::{Content, TensorInfo};
use std::{collections::BTreeSet, env, path::Path};

fn layer_number(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn moe_summary(content: &Content) -> Result<String, String> {
    let tensors: Vec<&TensorInfo> = content
        .tensors
        .iter()
        .filter(|tensor| tensor.name.ends_with("_exps.weight"))
        .collect();
    if tensors.is_empty() {
        return Err("no routed-expert tensors found".into());
    }
    let expert_counts: BTreeSet<u64> = tensors
        .iter()
        .filter_map(|tensor| tensor.dims.last().copied())
        .collect();
    if expert_counts.len() != 1 || expert_counts.contains(&0) {
        return Err(format!("inconsistent expert dimensions: {expert_counts:?}"));
    }
    let experts = *expert_counts.first().expect("one expert count");
    if tensors.iter().any(|tensor| tensor.len % experts != 0) {
        return Err("expert tensor bytes are not divisible by expert count".into());
    }
    let layers: BTreeSet<_> = tensors
        .iter()
        .filter_map(|tensor| layer_number(&tensor.name))
        .collect();
    if layers.is_empty() {
        return Err("expert tensors do not carry blk.N layer names".into());
    }
    let total_bytes: u64 = tensors.iter().map(|tensor| tensor.len).sum();
    let bytes_per_expert_all_layers = total_bytes / experts;
    let bytes_per_expert_layer = bytes_per_expert_all_layers
        / u64::try_from(layers.len()).map_err(|_| "layer count overflow")?;
    let metadata = content
        .metadata
        .iter()
        .filter(|(key, _)| key.contains("expert") || key.contains("moe"))
        .map(|(key, value)| format!("metadata\t{key}\t{value}"))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(format!(
        "moe\ttensors={}\tlayers={}\texperts={}\ttotal_bytes={}\tbytes_per_expert_layer={}\tbytes_per_expert_all_layers={}\n{}",
        tensors.len(),
        layers.len(),
        experts,
        total_bytes,
        bytes_per_expert_layer,
        bytes_per_expert_all_layers,
        metadata
    ))
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let (summary_only, path) = match args.as_slice() {
        [path] => (false, path),
        [flag, path] if flag == "--moe-summary" => (true, path),
        _ => {
            eprintln!("usage: spm-gguf-inspect [--moe-summary] <model.gguf>");
            std::process::exit(2);
        }
    };
    if path.is_empty() {
        eprintln!("usage: spm-gguf-inspect [--moe-summary] <model.gguf>");
        std::process::exit(2);
    }
    match spm_gguf::read(Path::new(&path)) {
        Ok(c) => {
            if summary_only {
                match moe_summary(&c) {
                    Ok(summary) => println!("{summary}"),
                    Err(error) => {
                        eprintln!("spm-gguf-inspect: {error}");
                        std::process::exit(1);
                    }
                }
                return;
            }
            println!(
                "version={} metadata={} tensors={} data_offset={}",
                c.version,
                c.metadata_count,
                c.tensors.len(),
                c.tensor_data_offset
            );
            for key in [
                "general.architecture",
                "qwen3.context_length",
                "general.alignment",
            ] {
                if let Some(v) = c.metadata.get(key) {
                    println!("{key}={v}");
                }
            }
            for t in &c.tensors {
                println!(
                    "tensor\t{}\t{:?}\ttype={}\toffset={}\tbytes={}",
                    t.name, t.dims, t.dtype, t.offset, t.len
                );
            }
        }
        Err(e) => {
            eprintln!("spm-gguf-inspect: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn summary_derives_expert_bytes_without_reading_payloads() {
        let tensors = (0..2)
            .flat_map(|layer| {
                ["up", "down"].map(move |kind| TensorInfo {
                    name: format!("blk.{layer}.ffn_{kind}_exps.weight"),
                    dims: vec![4, 8, 16],
                    dtype: 14,
                    offset: 0,
                    len: 1_600,
                })
            })
            .collect();
        let content = Content {
            version: 3,
            metadata_count: 0,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors,
        };
        let summary = moe_summary(&content).expect("valid summary");
        assert!(summary.contains("layers=2\texperts=16\ttotal_bytes=6400"));
        assert!(summary.contains("bytes_per_expert_layer=200"));
        assert!(summary.contains("bytes_per_expert_all_layers=400"));
    }
}
