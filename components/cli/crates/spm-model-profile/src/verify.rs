use crate::{digest, manifest::Envelope};
use std::{fs, path::Path};

pub fn run(manifest: &Path, model: &Path, runtime: &str, hardware: &Path) -> Result<(), String> {
    let bytes = fs::read(manifest).map_err(|error| error.to_string())?;
    let envelope: Envelope = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if envelope.schema != "emufpga.install-manifest-envelope.v1" {
        return Err("unsupported manifest schema".into());
    }
    let payload = serde_json::to_vec(&envelope.payload).map_err(|error| error.to_string())?;
    if digest::bytes(&payload) != envelope.payload_sha256 {
        return Err("manifest payload checksum mismatch".into());
    }
    let expected_runtime = field(&envelope, "/runtime_build")?;
    let expected_hardware = field(&envelope, "/hardware_sha256")?;
    if runtime != expected_runtime {
        return Err("stale manifest: runtime build changed".into());
    }
    if digest::file(hardware)? != expected_hardware {
        return Err("stale manifest: hardware topology changed".into());
    }
    let expected_model = field(&envelope, "/model/sha256")?;
    if digest::file(model)? != expected_model {
        return Err("stale manifest: model identity changed".into());
    }
    Ok(())
}

fn field<'a>(envelope: &'a Envelope, pointer: &str) -> Result<&'a str, String> {
    envelope
        .payload
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("missing manifest field {pointer}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_manifest(
        name: &str,
        payload: serde_json::Value,
        valid_checksum: bool,
    ) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("spm-profile-{name}-{}.json", std::process::id()));
        let digest = digest::bytes(&serde_json::to_vec(&payload).unwrap());
        let envelope = Envelope {
            schema: "emufpga.install-manifest-envelope.v1".into(),
            payload_sha256: if valid_checksum { digest } else { "bad".into() },
            payload,
        };
        fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
        path
    }

    #[test]
    fn rejects_tampered_payload() {
        let path = write_manifest("tamper", json!({}), false);
        assert!(
            run(&path, Path::new("missing"), "x", Path::new("missing"))
                .unwrap_err()
                .contains("checksum")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_stale_runtime_before_large_model_read() {
        let payload = json!({"runtime_build":"old","hardware_sha256":"x","model":{"sha256":"x"}});
        let path = write_manifest("runtime", payload, true);
        assert!(
            run(
                &path,
                Path::new("missing-model"),
                "new",
                Path::new("missing-hardware")
            )
            .unwrap_err()
            .contains("runtime")
        );
        fs::remove_file(path).unwrap();
    }
}
