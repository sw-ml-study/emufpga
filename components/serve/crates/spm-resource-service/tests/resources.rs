use std::{fs, process::Command};

fn paths(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base = std::env::temp_dir().join(format!("spm-resource-{}-{name}", std::process::id()));
    (base.with_extension("json"), base.with_extension("bin"))
}

#[test]
fn verifies_a_content_addressed_json_resource() {
    let (manifest, payload) = paths("valid");
    fs::write(&payload, b"abc").unwrap();
    let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let json = format!(
        r#"{{"schema":"emufpga.resource.v1","id":"sha256:{hash}","kind":"expert_stream","bytes":3,"sha256":"{hash}","tier":"hdd","mutable":false,"residency_limit_bytes":1024}}"#
    );
    fs::write(&manifest, json).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_spm-resource-service"))
        .args([
            "verify",
            manifest.to_str().unwrap(),
            payload.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let _ = (fs::remove_file(manifest), fs::remove_file(payload));
}

#[test]
fn rejects_executable_or_non_json_serialization() {
    let (manifest, payload) = paths("pickle");
    fs::write(&payload, b"abc").unwrap();
    fs::write(&manifest, b"cos\nsystem\n(S'echo unsafe'\ntR.").unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_spm-resource-service"))
        .args([
            "verify",
            manifest.to_str().unwrap(),
            payload.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    let _ = (fs::remove_file(manifest), fs::remove_file(payload));
}
