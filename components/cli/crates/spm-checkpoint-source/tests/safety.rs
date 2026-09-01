use spm_checkpoint_source::open;
use std::{
    fs, process,
    time::{SystemTime, UNIX_EPOCH},
};

fn path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("spm-{label}-{}-{nonce}.safetensors", process::id()))
}

#[test]
fn rejects_a_header_length_beyond_the_file() {
    let file = path("header");
    fs::write(&file, 1_000_000u64.to_le_bytes()).unwrap();
    let error = open(&file).unwrap_err();
    fs::remove_file(file).unwrap();
    assert!(error.contains("header exceeds file"));
}

#[test]
fn rejects_tensor_offsets_beyond_the_file() {
    let file = path("offset");
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    fs::write(&file, bytes).unwrap();
    let error = open(&file).unwrap_err();
    fs::remove_file(file).unwrap();
    assert!(error.contains("data exceeds checkpoint"));
}
