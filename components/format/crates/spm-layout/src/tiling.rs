//! The mapping from a logical matrix position to a stream position,
//! and the scale-group arithmetic over that stream.

/// Total weights in an `rows` by `cols` matrix.
#[must_use]
pub const fn weight_count(rows: u32, cols: u32) -> u64 {
    rows as u64 * cols as u64
}

/// Stream position of `W[row][col]`.
///
/// Column-major: consecutive positions walk down a column, so the
/// engine holds one activation while a whole column streams past. The
/// inverse is `row = index % rows`, `col = index / rows`, which is
/// cheap enough that consumers compute it inline rather than calling
/// back into this crate.
#[must_use]
pub const fn stream_index(rows: u32, row: u32, col: u32) -> u64 {
    col as u64 * rows as u64 + row as u64
}

/// Number of scale groups covering `total` weights.
///
/// The last group is short when `group_size` does not divide `total`.
#[must_use]
pub const fn group_count(total: u64, group_size: u32) -> u64 {
    if group_size == 0 {
        return 0;
    }
    total.div_ceil(group_size as u64)
}

/// Weights in group `group`, which is short if it is the last one.
///
/// Returns zero for a group index past the end.
#[must_use]
pub fn group_len(total: u64, group_size: u32, group: u64) -> u32 {
    if group_size == 0 {
        return 0;
    }
    let start = u64::from(group_size) * group;
    if start >= total {
        return 0;
    }
    // A remainder wider than u32 is necessarily larger than
    // group_size, so saturating to group_size is the right answer and
    // no narrowing cast is needed.
    u32::try_from(total - start)
        .unwrap_or(group_size)
        .min(group_size)
}
