//! The recursion, and the rewind that makes it a rotating store.

use crate::block::Layer;
use crate::config::TrmConfig;
use spm_linear::LinearError;
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;
use spm_stream_metrics::widen;

/// What one forward pass consumed.
///
/// Counts rather than timings: this step establishes mechanism, and
/// docs/results.md carries the measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Forward {
    /// `L_level` calls executed.
    pub calls: usize,
    /// Rewinds issued -- one before each call after the first.
    pub rewinds: usize,
    /// Weights read off the stream, counting re-reads.
    pub weights_read: u64,
    /// Distinct weights in the rotating region.
    pub weights_distinct: u64,
    /// Batch positions carried through every sweep.
    pub positions: usize,
}

impl Forward {
    /// Scan productivity counting **both** reuse axes.
    ///
    /// `Ps` as `spm-stream-metrics` defines it counts applications per
    /// weight read, and a scan-level view sees only batch reuse. For a
    /// recursive model that under-reports badly: TRM re-reads its
    /// whole rotating region 15 times per forward, so the weights are
    /// reused across *depth* as well as across batch.
    ///
    /// This counts the distinct weights actually held, so the answer
    /// is `positions * calls` -- 15x higher than a per-scan view, and
    /// the honest number for a model that rotates.
    #[must_use]
    pub fn scan_productivity(&self) -> Option<f64> {
        (self.weights_distinct > 0).then(|| {
            let positions = u64::try_from(self.positions).unwrap_or(u64::MAX);
            widen(self.weights_read * positions) / widen(self.weights_distinct)
        })
    }
}

/// Runs `config.level_calls()` sweeps of the rotating region.
///
/// Each call rewinds first, except the first, so every sweep starts at
/// stream 0 -- which is why `layouts/*.order` puts the rotating region
/// there. `rewind` is legal here because it happens *between*
/// operations: an `L_level` call is one operation, and nothing seeks
/// inside one.
///
/// The two latent states are the caller's; they stay resident and are
/// kilobytes against megabytes of weights.
///
/// # Errors
/// Returns [`LinearError`] if a sweep ends early, or the stream fails.
pub fn forward<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &TrmConfig,
    state: &mut [f32],
    layers: &mut [Layer],
) -> Result<Forward, LinearError> {
    let positions = state.len() / config.hidden;
    let mut report = Forward {
        positions,
        weights_distinct: distinct_weights(config, groups),
        ..Forward::default()
    };
    for call in 0..config.level_calls() {
        if call > 0 {
            rewind(groups)?;
            report.rewinds += 1;
        }
        for layer in layers.iter_mut() {
            layer.forward(groups, config, state)?;
        }
        report.calls += 1;
        report.weights_read += report.weights_distinct;
    }
    Ok(report)
}

/// Returns the stream to its first group, between operations.
fn rewind<S: WeightStream>(groups: &mut GroupStream<S>) -> Result<(), LinearError> {
    groups.rewind().map_err(|e| LinearError::Stream {
        detail: e.to_string(),
    })
}

/// Weights in the rotating region, from the stream directory.
fn distinct_weights<S: WeightStream>(config: &TrmConfig, groups: &GroupStream<S>) -> u64 {
    groups
        .descriptors
        .iter()
        .take(config.layers * 4)
        .map(|d| u64::from(d.rows) * u64::from(d.cols))
        .sum()
}
