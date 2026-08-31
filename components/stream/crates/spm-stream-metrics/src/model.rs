//! Raw counters gathered over one parameter scan.

use std::time::Duration;

/// Counters describing a single pass over a parameter stream.
///
/// Populated by whoever drives the scan; the derived metrics live in
/// sibling modules.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScanMetrics {
    /// Parameter bytes pulled off the backing store.
    pub parameter_bytes_read: u64,
    /// Weights unpacked from those bytes.
    pub weights_decoded: u64,
    /// Useful parameter applications. One weight applied to one
    /// activation is one application, so a batch of 32 turns each
    /// decoded weight into 32 applications.
    pub weight_applications: u64,
    /// Parameter bytes held in random-access memory at any moment.
    pub resident_parameter_bytes: u64,
    /// Total parameter bytes in the model being scanned.
    pub total_parameter_bytes: u64,
    /// Time spent waiting on the backing store.
    pub storage_time: Duration,
    /// Time spent in the tensor engine.
    pub compute_time: Duration,
}

impl ScanMetrics {
    /// Wall-clock time for the scan.
    ///
    /// Storage and compute partition wall clock only while IO is not
    /// overlapped, which is the case today (see `spm-stream-file`).
    /// Once a prefetching backend lands they will overlap, and this
    /// sum becomes an upper bound rather than the elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.storage_time + self.compute_time
    }
}

/// Widens a `u64` to `f64` without a lint-suppressing cast.
///
/// Public because a second crate needs it: `fabric-model` computes the
/// same kind of ratio over cycle counts. One definition with the
/// reasoning attached beats the same three lines copied with the
/// reasoning lost.
///
/// Rust has no `From<u64> for f64` because the conversion is lossy
/// above 2^53, and `as` trips `clippy::cast_precision_loss`. Blanket
/// allowing that lint across this crate would also hide genuine
/// precision bugs in the ratio arithmetic, so the split is done
/// explicitly instead. Both `try_from` calls are infallible by
/// construction: a `u64` shifted right by 32, and a `u64` masked to
/// its low 32 bits, each fit a `u32`.
#[must_use]
pub fn widen(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high).mul_add(4_294_967_296.0, f64::from(low))
}
