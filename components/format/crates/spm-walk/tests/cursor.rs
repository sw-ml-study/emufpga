//! The cursor must visit every group of every stream, exactly once,
//! in order -- and skip streams that declare no weights.

use spm_layout::{Encoding, OpDescriptor};
use spm_walk::Cursor;

fn descriptor(rows: u32, cols: u32, group_size: u32) -> OpDescriptor {
    OpDescriptor {
        rows,
        cols,
        group_size,
        encoding: Encoding::Ternary2F32I32,
        lane_count: 1,
    }
}

fn visit(descriptors: &[OpDescriptor]) -> Vec<(usize, u64, u32)> {
    let mut cursor = Cursor::new(descriptors);
    let mut seen = Vec::new();
    while let Some(len) = cursor.group_len(descriptors) {
        seen.push((cursor.stream, cursor.group, len));
        cursor.advance(descriptors);
    }
    seen
}

#[test]
fn visits_every_group_of_every_stream_in_order() {
    // 6 weights at group 4 -> groups of 4 then 2; then 4 weights at
    // group 2 -> two full groups.
    let descriptors = [descriptor(3, 2, 4), descriptor(2, 2, 2)];
    assert_eq!(
        visit(&descriptors),
        vec![(0, 0, 4), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    );
}

#[test]
fn streams_declaring_no_weights_are_skipped() {
    // A zero-weight stream has no groups at all. If the cursor rested
    // on one, the reader would try to consume a scale that was never
    // written and read the next stream's bytes as its own.
    let descriptors = [
        descriptor(0, 5, 4),
        descriptor(2, 1, 4),
        descriptor(3, 0, 4),
    ];
    assert_eq!(visit(&descriptors), vec![(1, 0, 2)]);
}

#[test]
fn an_empty_directory_yields_nothing() {
    assert_eq!(visit(&[]), Vec::new());
    let mut cursor = Cursor::new(&[]);
    cursor.advance(&[]);
    assert_eq!(cursor.group_len(&[]), None);
}
