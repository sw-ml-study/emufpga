//! Consumption order and scale-group arithmetic.

use spm_layout::{group_count, group_len, stream_index, weight_count};

#[test]
fn stream_order_is_column_major() {
    // Consecutive stream positions walk DOWN a column, so the engine
    // holds one activation while a whole column streams past. This is
    // the entire reason the layout exists.
    let rows = 3;
    assert_eq!(stream_index(rows, 0, 0), 0);
    assert_eq!(stream_index(rows, 1, 0), 1);
    assert_eq!(stream_index(rows, 2, 0), 2);
    assert_eq!(stream_index(rows, 0, 1), 3);
    assert_eq!(stream_index(rows, 2, 1), 5);
}

#[test]
fn the_documented_inverse_mapping_holds() {
    // Consumers compute row and col inline rather than calling back
    // into this crate, so the identity is pinned here.
    let (rows, cols) = (7u32, 5u32);
    for col in 0..cols {
        for row in 0..rows {
            let index = stream_index(rows, row, col);
            assert_eq!(index % u64::from(rows), u64::from(row));
            assert_eq!(index / u64::from(rows), u64::from(col));
        }
    }
    assert_eq!(weight_count(rows, cols), 35);
}

#[test]
fn a_group_size_that_does_not_divide_leaves_a_short_final_group() {
    let total = weight_count(3, 2); // 6 weights
    assert_eq!(group_count(total, 4), 2);
    assert_eq!(group_len(total, 4, 0), 4);
    assert_eq!(group_len(total, 4, 1), 2);
    assert_eq!(group_len(total, 4, 2), 0);
}

#[test]
fn group_arithmetic_survives_degenerate_inputs() {
    assert_eq!(group_count(0, 64), 0);
    assert_eq!(group_len(0, 64, 0), 0);
    assert_eq!(group_count(10, 0), 0);
    assert_eq!(group_len(10, 0, 0), 0);
    // One scale per column is the case that lets the engine pre-scale
    // the activation and keep the inner loop multiplier-free.
    let total = weight_count(8, 4);
    assert_eq!(group_count(total, 8), 4);
    assert_eq!(group_len(total, 8, 3), 8);
}
