use spm_checkpoint_source::{DType, open, read_tensor};
use std::path::Path;

#[test]
fn reads_the_pytorch_subset_used_by_real_checkpoints() {
    let path = Path::new("tests/fixtures/tiny.pt");
    let mut tensors = open(path).expect("fixture should parse");
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].name, "bias");
    assert_eq!(tensors[0].shape, [3]);
    assert_eq!(tensors[1].name, "weight");
    assert_eq!(tensors[1].shape, [3, 2]);
    assert!(tensors.iter().all(|tensor| tensor.dtype == DType::F32));
    let raw = read_tensor(&tensors[1]).expect("tensor bytes should read");
    let values = raw
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    assert_eq!(values, [0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
}
