use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, fs::File, io::Read, path::Path};

#[derive(Deserialize, Serialize)]
pub struct Manifest {
    pub schema: String,
    pub id: String,
    pub kind: Kind,
    pub bytes: u64,
    pub sha256: String,
    pub tier: Tier,
    pub mutable: bool,
    pub residency_limit_bytes: u64,
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Model,
    Tensor,
    ExpertStream,
    KvCache,
    Activation,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Hdd,
    Ssd,
    Pmem,
    Ram,
    Vram,
}

pub fn load(path: &Path) -> Result<Manifest, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

pub fn verify(manifest: &str, payload: &str) -> Result<(), String> {
    let resource = load(Path::new(manifest))?;
    validate(&resource)?;
    let actual = payload_digest(Path::new(payload))?;
    if actual != (resource.bytes, resource.sha256.clone()) {
        return Err("payload does not match content address".into());
    }
    println!("{{\"id\":\"{}\",\"verified\":true}}", resource.id);
    Ok(())
}

pub fn payload_digest(path: &Path) -> Result<(u64, String), String> {
    let mut source = File::open(path).map_err(|error| error.to_string())?;
    let (mut bytes, mut hash) = (0_u64, Sha256::new());
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = source
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            return Ok((bytes, format!("{:x}", hash.finalize())));
        }
        hash.update(&buffer[..count]);
        bytes += count as u64;
    }
}

fn validate(resource: &Manifest) -> Result<(), String> {
    let expected_id = format!("sha256:{}", resource.sha256);
    if resource.schema != "emufpga.resource.v1" || resource.id != expected_id {
        return Err("unsupported or malformed manifest".into());
    }
    let state = matches!(resource.kind, Kind::KvCache | Kind::Activation);
    if state != resource.mutable || resource.residency_limit_bytes == 0 {
        return Err("invalid mutability or residency contract".into());
    }
    Ok(())
}
