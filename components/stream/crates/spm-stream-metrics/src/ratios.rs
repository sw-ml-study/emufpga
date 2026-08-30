//! Scan productivity and parameter residency: the economic argument.

use crate::model::{ScanMetrics, widen};

impl ScanMetrics {
    /// `Ps` -- useful parameter applications per parameter value read.
    ///
    /// Batch-1 dense inference gives `Ps == 1`: every weight is read,
    /// used once, and thrown away. Every technique the research
    /// proposes exists to raise this number -- batching to `~batch`,
    /// `MoE` scheduling to the queue depth per expert, speculative
    /// decoding to the accepted-token count -- because storage
    /// bandwidth is fixed and reuse is the only free variable.
    ///
    /// `None` if no weights were read.
    #[must_use]
    pub fn scan_productivity(&self) -> Option<f64> {
        (self.weights_decoded > 0)
            .then(|| widen(self.weight_applications) / widen(self.weights_decoded))
    }

    /// `Rp` -- parameter bytes resident in RAM over total parameter
    /// bytes.
    ///
    /// Conventional inference sits at `Rp ~= 1`: the model is loaded.
    /// The goal is `Rp -> 0` while activation and state memory stay
    /// nonzero. A group-at-a-time streamer holds one group, so `Rp` is
    /// small but never zero, and this reports that honestly rather
    /// than rounding a real buffer down to nothing.
    ///
    /// `None` if the model has no parameters.
    #[must_use]
    pub fn residency(&self) -> Option<f64> {
        (self.total_parameter_bytes > 0)
            .then(|| widen(self.resident_parameter_bytes) / widen(self.total_parameter_bytes))
    }
}
