//! What a forward pass consumed.

use spm_stream_metrics::widen;

/// What one HRM forward pass consumed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HrmForward {
    /// Low-level module sweeps.
    pub low_sweeps: usize,
    /// High-level module sweeps.
    pub high_sweeps: usize,
    /// Rewinds issued.
    pub rewinds: usize,
    /// Weights read, counting re-reads.
    pub weights_read: u64,
    /// Distinct weights in the rotating region.
    pub weights_distinct: u64,
    /// Batch positions carried through every sweep.
    pub positions: usize,
}

impl HrmForward {
    /// An empty report for a run of `positions` over `distinct`
    /// weights.
    #[must_use]
    pub fn new(positions: usize, distinct: u64) -> Self {
        Self {
            positions,
            weights_distinct: distinct,
            ..Self::default()
        }
    }

    /// Scan productivity counting recursion as well as batch.
    ///
    /// Same reasoning as TRM: a scan-level view of `Ps` sees batch
    /// reuse only, and a recursive model reuses across depth too.
    #[must_use]
    pub fn scan_productivity(&self) -> Option<f64> {
        (self.weights_distinct > 0).then(|| {
            let positions = u64::try_from(self.positions).unwrap_or(u64::MAX);
            widen(self.weights_read * positions) / widen(self.weights_distinct)
        })
    }
}
