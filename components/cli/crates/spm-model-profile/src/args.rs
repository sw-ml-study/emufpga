use crate::{digest, inventory, manifest, profile, verify};
use serde_json::Value;
use std::{fs, path::PathBuf};

struct Install {
    model: PathBuf,
    output: PathBuf,
    runtime: String,
    hardware: PathBuf,
    traces: Vec<PathBuf>,
    ram_mib: u64,
    vram_mib: u64,
}

pub fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("install" | "refresh") => install(&parse_install(&args[1..])?),
        Some("verify") => verify_args(&args[1..]),
        _ => Err(usage()),
    }
}

fn install(config: &Install) -> Result<(), String> {
    let content = spm_gguf::read(&config.model)?;
    let inventory = inventory::build(&content)?;
    let workload = profile::build(&config.traces)?;
    let model_sha = digest::file(&config.model)?;
    let hardware_bytes = fs::read(&config.hardware).map_err(|error| error.to_string())?;
    let hardware: Value =
        serde_json::from_slice(&hardware_bytes).map_err(|error| error.to_string())?;
    let hardware_sha = digest::bytes(&hardware_bytes);
    let inputs = manifest::Inputs {
        model_path: &config.model,
        model_sha: &model_sha,
        runtime: &config.runtime,
        hardware: &hardware,
        hardware_sha: &hardware_sha,
        ram_mib: config.ram_mib,
        vram_mib: config.vram_mib,
    };
    manifest::write(&config.output, &content, &inventory, &workload, &inputs)
}

fn parse_install(args: &[String]) -> Result<Install, String> {
    let mut model = None;
    let mut output = None;
    let mut runtime = None;
    let mut hardware = None;
    let mut traces = Vec::new();
    let mut ram_mib = None;
    let mut vram_mib = None;
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1).ok_or_else(usage)?;
        match args[index].as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--runtime" => runtime = Some(value.clone()),
            "--hardware" => hardware = Some(PathBuf::from(value)),
            "--trace" => traces.push(PathBuf::from(value)),
            "--ram-mib" => ram_mib = Some(number(value)?),
            "--vram-mib" => vram_mib = Some(number(value)?),
            flag => return Err(format!("unknown install option {flag}")),
        }
        index += 2;
    }
    Ok(Install {
        model: model.ok_or_else(usage)?,
        output: output.ok_or_else(usage)?,
        runtime: runtime.ok_or_else(usage)?,
        hardware: hardware.ok_or_else(usage)?,
        traces,
        ram_mib: ram_mib.ok_or_else(usage)?,
        vram_mib: vram_mib.ok_or_else(usage)?,
    })
}

fn verify_args(args: &[String]) -> Result<(), String> {
    if args.len() != 4 {
        return Err(usage());
    }
    verify::run(
        &PathBuf::from(&args[0]),
        &PathBuf::from(&args[1]),
        &args[2],
        &PathBuf::from(&args[3]),
    )
}

fn number(value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid nonnegative integer: {value}"))
}

fn usage() -> String {
    "usage: spm-model-profile install|refresh --model M --output O --runtime ID --hardware H.json --ram-mib N --vram-mib N [--trace LOG]... | verify MANIFEST MODEL RUNTIME HARDWARE.json".into()
}
