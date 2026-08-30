//! Throughput derived from the raw counters.

use crate::model::{ScanMetrics, widen};

impl ScanMetrics {
    /// Bytes per second the backing store actually delivered.
    ///
    /// Measured against storage time alone, so it reports what the
    /// store can do rather than what the pipeline achieved. `None` if
    /// no time was spent on storage.
    #[must_use]
    pub fn raw_bandwidth(&self) -> Option<f64> {
        let seconds = self.storage_time.as_secs_f64();
        (seconds > 0.0).then(|| widen(self.parameter_bytes_read) / seconds)
    }

    /// Weights unpacked per second of wall clock.
    ///
    /// Compared against [`ScanMetrics::useful_weights_per_sec`], the
    /// gap between the two is exactly the reuse the architecture is
    /// buying.
    #[must_use]
    pub fn decoded_weights_per_sec(&self) -> Option<f64> {
        let seconds = self.elapsed().as_secs_f64();
        (seconds > 0.0).then(|| widen(self.weights_decoded) / seconds)
    }

    /// Useful parameter applications per second of wall clock.
    #[must_use]
    pub fn useful_weights_per_sec(&self) -> Option<f64> {
        let seconds = self.elapsed().as_secs_f64();
        (seconds > 0.0).then(|| widen(self.weight_applications) / seconds)
    }

    /// Engine consumption bandwidth over storage bandwidth.
    ///
    /// Both are the same byte count over a different time, so this
    /// reduces to `storage_time / compute_time`.
    ///
    /// - `eta >= 1` -- the engine consumes at least as fast as storage
    ///   supplies, so the scan is storage-bound. That is the regime
    ///   the architecture is designed for.
    /// - `eta < 1` -- compute is the bottleneck, and the hardware
    ///   design has a concrete target: make the tensor engine `1/eta`
    ///   times faster, or widen the lanes.
    ///
    /// `None` if no time was spent computing.
    #[must_use]
    pub fn eta(&self) -> Option<f64> {
        let compute = self.compute_time.as_secs_f64();
        (compute > 0.0).then(|| self.storage_time.as_secs_f64() / compute)
    }
}
