use std::{fs, path::PathBuf};

fn temp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("spm-gguf-{name}-{}", std::process::id()));
    fs::write(&p, bytes).unwrap();
    p
}
fn header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut v = b"GGUF".to_vec();
    v.extend(3u32.to_le_bytes());
    v.extend(tensors.to_le_bytes());
    v.extend(metadata.to_le_bytes());
    v
}

#[test]
fn rejects_bad_magic() {
    let p = temp("magic", b"NOPE");
    assert!(spm_gguf::read(&p).unwrap_err().contains("magic"));
    fs::remove_file(p).unwrap();
}
#[test]
fn rejects_unbounded_counts() {
    let p = temp("count", &header(0, spm_gguf::MAX_METADATA + 1));
    assert!(spm_gguf::read(&p).unwrap_err().contains("metadata count"));
    fs::remove_file(p).unwrap();
}
#[test]
fn rejects_unbounded_strings_before_allocation() {
    let mut v = header(0, 1);
    v.extend((spm_gguf::MAX_STRING + 1).to_le_bytes());
    let p = temp("string", &v);
    assert!(spm_gguf::read(&p).unwrap_err().contains("string length"));
    fs::remove_file(p).unwrap();
}

#[test]
fn rejects_tensor_ranges_beyond_the_file() {
    let mut v = header(1, 0);
    v.extend(1u64.to_le_bytes());
    v.push(b'x');
    v.extend(1u32.to_le_bytes());
    v.extend(32u64.to_le_bytes());
    v.extend(0u32.to_le_bytes());
    v.extend(4096u64.to_le_bytes());
    v.resize(64, 0);
    let p = temp("range", &v);
    assert!(spm_gguf::read(&p).unwrap_err().contains("beyond file"));
    fs::remove_file(p).unwrap();
}
#[test]
fn reads_a_tiny_aligned_fixture() {
    let mut v = header(1, 0);
    v.extend(1u64.to_le_bytes());
    v.push(b'x');
    v.extend(1u32.to_le_bytes());
    v.extend(32u64.to_le_bytes());
    v.extend(0u32.to_le_bytes());
    v.extend(0u64.to_le_bytes());
    v.resize(64, 0);
    v.resize(64 + 32 * 4, 1);
    let p = temp("tiny", &v);
    let c = spm_gguf::read(&p).unwrap();
    assert_eq!(c.tensors[0].len, 128);
    assert_eq!(c.tensors[0].offset, 64);
    assert_eq!(
        spm_gguf::read_tensor_range(&p, &c.tensors[0], 3, 4, 4).unwrap(),
        [1, 1, 1, 1]
    );
    assert!(spm_gguf::read_tensor_range(&p, &c.tensors[0], 127, 2, 2).is_err());
    assert!(spm_gguf::read_tensor_range(&p, &c.tensors[0], 0, 5, 4).is_err());
    fs::remove_file(p).unwrap();
}
