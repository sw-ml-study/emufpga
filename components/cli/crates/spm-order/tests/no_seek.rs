//! A forward pass must never need a stream index that goes backward.
//!
//! Hermetic against the REAL order file. `layouts/` is tracked text,
//! so the ordering can be tested without the 27 MB checkpoint -- the
//! weights would be needed to test the values, but not the layout.

use spm_import::Tensor;
use spm_order::{apply_order, parse_order, reorder};
use std::path::Path;

/// The shipped TRM order.
fn trm_order_text() -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../layouts/trm-maze-30x30.order");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn tensor(name: &str) -> Tensor {
    Tensor {
        name: name.into(),
        shape: vec![4, 4],
        blob: "x.bin".into(),
    }
}

/// The order a TRM forward pass consumes weights in, from `trm.py`:
/// 15 `L_level` calls, each sweeping both layers, each layer running
/// attention then MLP.
fn forward_pass_demands() -> Vec<String> {
    let mut wanted = Vec::new();
    for _call in 0..15 {
        for layer in 0..2 {
            for part in [
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj",
            ] {
                wanted.push(format!(
                    "_orig_mod.model.inner.L_level.layers.{layer}.{part}.weight"
                ));
            }
        }
    }
    wanted
}

#[test]
fn a_forward_sweep_never_moves_backward() {
    // THE POINT OF THE STEP. Walk what a forward pass asks for, in
    // order, and check the stream index only ever increases -- resetting
    // to zero exactly at a rewind, never mid-operation.
    let order = parse_order(&trm_order_text()).expect("parse order");
    let index = |name: &str| {
        order
            .rotating
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("{name} is not in the rotating region"))
    };

    let demands = forward_pass_demands();
    let rotating = order.rotating.len();
    let mut previous = None;
    let mut rewinds = 0;
    for name in &demands {
        let at = index(name);
        if let Some(prev) = previous
            && at <= prev
        {
            // Only legal as a rewind to the very start of the region.
            assert_eq!(at, 0, "sweep went backward to {at} from {prev}: {name}");
            assert_eq!(prev, rotating - 1, "rewound before finishing the sweep");
            rewinds += 1;
        }
        previous = Some(at);
    }
    // One L_level call sweeps BOTH layers, so it consumes the whole
    // 8-stream region once. 15 calls is 15 sweeps and 14 rewinds --
    // derived rather than written down, so the expectation cannot
    // drift from the demand list the way a hardcoded count can.
    assert_eq!(demands.len(), 120);
    let sweeps = demands.len() / rotating;
    assert_eq!(
        sweeps, 15,
        "15 L_level calls, each sweeping the region once"
    );
    assert_eq!(rewinds, sweeps - 1, "one rewind between consecutive sweeps");
}

#[test]
fn the_rotating_region_comes_first() {
    // rewind() returns to stream 0, so anything ahead of the rotating
    // region would be re-read on every sweep for nothing.
    let order = parse_order(&trm_order_text()).expect("parse");
    assert_eq!(order.rotating.len(), 8);
    assert_eq!(order.resident.len(), 7);
    let tensors: Vec<Tensor> = order
        .rotating
        .iter()
        .chain(&order.resident)
        .map(|n| tensor(n))
        .collect();
    let ordered = reorder(&tensors, &order).expect("reorder");
    for (i, name) in order.rotating.iter().enumerate() {
        assert_eq!(&ordered[i].name, name, "rotating stream {i} out of place");
    }
}

#[test]
fn alphabetical_order_would_have_seeked_backward() {
    // What step 002 shipped, kept as a test so the mistake cannot
    // quietly return. Sorting the rotating names alphabetically gives
    // down, gate_up, o, qkv -- the reverse of execution order.
    let order = parse_order(&trm_order_text()).expect("parse");
    let mut alphabetical = order.rotating.clone();
    alphabetical.sort();
    assert_ne!(
        alphabetical, order.rotating,
        "alphabetical must differ from consumption order, or this test proves nothing"
    );
    // Under the alphabetical layout, the first thing a forward pass
    // wants (qkv_proj) sits at index 3 and the second (o_proj) at 2.
    let qkv = alphabetical
        .iter()
        .position(|n| n.contains("qkv_proj"))
        .expect("qkv");
    let o = alphabetical
        .iter()
        .position(|n| n.contains("o_proj"))
        .expect("o");
    assert!(
        o < qkv,
        "alphabetical puts o_proj before qkv_proj, so a sweep seeks back"
    );
}

#[test]
fn a_layout_that_disagrees_with_the_checkpoint_is_refused() {
    // Both directions. Silence on either is how a layout drifts away
    // from the model it describes.
    let order = parse_order("[rotating]\na\nb\n").expect("parse");
    let missing = reorder(&[tensor("a")], &order).expect_err("b is missing");
    assert!(format!("{missing}").contains("not in the checkpoint"));

    let extra =
        reorder(&[tensor("a"), tensor("b"), tensor("c")], &order).expect_err("c is unlisted");
    assert!(format!("{extra}").contains("not named in the order file"));
}

#[test]
fn no_order_file_means_no_rotating_region() {
    // Honest rather than convenient: a caller who supplied no order
    // should not be told they have a rotating region they do not have.
    let tensors = vec![tensor("a"), tensor("b")];
    let (out, rotating) = apply_order(tensors.clone(), None).expect("no order");
    assert_eq!(out, tensors);
    assert_eq!(rotating, 0);
}
