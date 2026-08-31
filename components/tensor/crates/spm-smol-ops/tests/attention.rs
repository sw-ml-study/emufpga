//! The two details that differ from every earlier rung.

use spm_smol_ops::grouped_causal;

#[test]
fn the_causal_mask_includes_the_diagonal() {
    // BDH's mask is .tril(diagonal=-1) and excludes it; Llama's
    // includes it. With one position there is exactly one key, so the
    // softmax puts all its weight there and the output is v[0]. Under
    // BDH's convention position 0 would attend to nothing and the row
    // would be zero (or NaN), so this discriminates.
    let (heads, kv_heads, head_dim) = (1, 1, 2);
    let q = vec![1.0f32, 0.0];
    let k = vec![1.0f32, 0.0];
    let v = vec![7.0f32, -3.0];
    let mut out = vec![0.0f32; 2];
    grouped_causal(&q, (&k, &v), (1, heads, kv_heads, head_dim), &mut out);
    assert_eq!(out, v, "position 0 must attend to itself");
}

#[test]
fn query_heads_share_their_key_value_head() {
    // 4 query heads over 2 KV heads: heads 0,1 read KV head 0 and
    // heads 2,3 read KV head 1. Values are chosen so each KV head is
    // identifiable in the output.
    let (positions, heads, kv_heads, head_dim) = (1, 4, 2, 2);
    let q = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let k = vec![1.0f32, 0.0, 1.0, 0.0];
    let v = vec![5.0f32, 5.0, 9.0, 9.0];
    let mut out = vec![0.0f32; positions * heads * head_dim];
    grouped_causal(
        &q,
        (&k, &v),
        (positions, heads, kv_heads, head_dim),
        &mut out,
    );
    assert_eq!(&out[0..2], &[5.0, 5.0], "head 0 -> kv head 0");
    assert_eq!(&out[2..4], &[5.0, 5.0], "head 1 -> kv head 0");
    assert_eq!(&out[4..6], &[9.0, 9.0], "head 2 -> kv head 1");
    assert_eq!(&out[6..8], &[9.0, 9.0], "head 3 -> kv head 1");
}

#[test]
fn a_later_position_cannot_see_an_earlier_one_only_the_reverse() {
    // The causal property itself. Position 0's output must not change
    // when position 1's value changes; position 1's must.
    let shape = (2, 1, 1, 2);
    let q = vec![1.0f32, 0.0, 1.0, 0.0];
    let k = vec![1.0f32, 0.0, 1.0, 0.0];
    let mut first = vec![0.0f32; 4];
    let mut second = vec![0.0f32; 4];
    grouped_causal(&q, (&k, &[1.0, 1.0, 2.0, 2.0]), shape, &mut first);
    grouped_causal(&q, (&k, &[1.0, 1.0, 99.0, 99.0]), shape, &mut second);
    assert_eq!(&first[0..2], &second[0..2], "position 0 saw the future");
    assert_ne!(
        &first[2..4],
        &second[2..4],
        "position 1 did not see position 1"
    );
}
