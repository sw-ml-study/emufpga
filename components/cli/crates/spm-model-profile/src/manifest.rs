use crate::{digest, inventory, profile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use spm_gguf::Content;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Serialize, Deserialize)]
pub struct Envelope {
    pub schema: String,
    pub payload_sha256: String,
    pub payload: Value,
}

#[derive(Serialize)]
struct Model<'a> {
    path: String,
    sha256: &'a str,
    bytes: u64,
    gguf_version: u32,
    architecture: &'a str,
    alignment: u64,
}

#[derive(Serialize)]
struct Placement {
    preload: Vec<String>,
    warm_candidates: Vec<String>,
    adaptive_priors: BTreeMap<String, u64>,
    fallback: String,
}

#[derive(Serialize)]
struct Payload<'a> {
    profile_version: &'a str,
    model: Model<'a>,
    runtime_build: &'a str,
    hardware: &'a Value,
    hardware_sha256: &'a str,
    ram_budget_bytes: u64,
    vram_budget_bytes: u64,
    inventory: &'a inventory::Inventory,
    workload: &'a profile::WorkloadProfile,
    placement: Placement,
}

pub struct Inputs<'a> {
    pub model_path: &'a Path,
    pub model_sha: &'a str,
    pub runtime: &'a str,
    pub hardware: &'a Value,
    pub hardware_sha: &'a str,
    pub ram_mib: u64,
    pub vram_mib: u64,
}

pub fn write(
    path: &Path,
    content: &Content,
    inventory: &inventory::Inventory,
    workload: &profile::WorkloadProfile,
    input: &Inputs<'_>,
) -> Result<(), String> {
    let payload = payload(content, inventory, workload, input)?;
    let encoded = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    let envelope = Envelope {
        schema: "emufpga.install-manifest-envelope.v1".into(),
        payload_sha256: digest::bytes(&encoded),
        payload,
    };
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&envelope).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(temporary, path).map_err(|error| error.to_string())
}

fn payload(
    content: &Content,
    inventory: &inventory::Inventory,
    workload: &profile::WorkloadProfile,
    input: &Inputs<'_>,
) -> Result<Value, String> {
    let architecture = content
        .metadata
        .get("general.architecture")
        .ok_or("GGUF has no architecture")?;
    let alignment = content
        .metadata
        .get("general.alignment")
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    serde_json::to_value(Payload {
        profile_version: "emufpga.install-profile.v1",
        model: Model {
            path: input.model_path.to_string_lossy().into_owned(),
            sha256: input.model_sha,
            bytes: fs::metadata(input.model_path)
                .map_err(|error| error.to_string())?
                .len(),
            gguf_version: content.version,
            architecture,
            alignment,
        },
        runtime_build: input.runtime,
        hardware: input.hardware,
        hardware_sha256: input.hardware_sha,
        ram_budget_bytes: input.ram_mib * 1_048_576,
        vram_budget_bytes: input.vram_mib * 1_048_576,
        inventory,
        workload,
        placement: placement(inventory, workload, input.ram_mib * 1_048_576),
    })
    .map_err(|error| error.to_string())
}

fn placement(
    inventory: &inventory::Inventory,
    workload: &profile::WorkloadProfile,
    budget: u64,
) -> Placement {
    let per_layer: BTreeMap<_, _> =
        inventory
            .tensors
            .iter()
            .fold(BTreeMap::new(), |mut sizes, tensor| {
                *sizes.entry(tensor.layer).or_insert(0_u64) += tensor.bytes_per_expert;
                sizes
            });
    let mut used = 0;
    let preload = workload
        .ranking
        .iter()
        .filter_map(|item| {
            let layer = item.resource.split(':').next()?.parse::<usize>().ok()?;
            let bytes = *per_layer.get(&layer)?;
            if used + bytes > budget / 2 {
                return None;
            }
            used += bytes;
            Some(item.resource.clone())
        })
        .collect::<Vec<_>>();
    let warm_candidates = workload
        .ranking
        .iter()
        .map(|item| item.resource.clone())
        .filter(|item| !preload.contains(item))
        .take(preload.len())
        .collect();
    let adaptive_priors = workload
        .ranking
        .iter()
        .map(|item| (item.resource.clone(), item.assignments))
        .collect();
    Placement {
        preload,
        warm_candidates,
        adaptive_priors,
        fallback: "unknown routes use bounded demand loading and are never rejected".into(),
    }
}
