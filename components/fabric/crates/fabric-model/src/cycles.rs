//! Cycle accounting for the fetch / FIFO / issue pipeline.

use crate::config::FabricConfig;
use spm_accum::AccumulatorBank;
use spm_stream_metrics::widen;

/// The pipeline's cycle state over one scan.
///
/// Deliberately coarse. It models one thing -- whether the datapath
/// has bytes to work on -- and does not pretend to model routing,
/// clock domains, or anything a real fabric would make you care about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pipeline {
    /// Total cycles elapsed.
    pub cycles: u64,
    /// Cycles the datapath spent waiting on an empty FIFO.
    pub stall_cycles: u64,
    /// Cycles the fetch stage spent blocked on a full FIFO.
    ///
    /// The mirror of `stall_cycles`. A run with many of these is
    /// compute-bound; a run with many stalls is fetch-bound. Both
    /// large means the FIFO is too small to decouple them.
    pub backpressure_cycles: u64,
    /// Bytes currently held in the FIFO.
    pub fifo_level: usize,
}

impl Pipeline {
    /// A pipeline that has paid its startup latency and nothing else.
    #[must_use]
    pub const fn start(config: &FabricConfig) -> Self {
        Self {
            cycles: config.fetch_latency_cycles,
            stall_cycles: 0,
            backpressure_cycles: 0,
            fifo_level: 0,
        }
    }

    /// Fetches one group into the FIFO, then issues it.
    ///
    /// One call rather than separate fetch and issue steps: the FIFO
    /// must hold a group before the datapath can start on it, and a
    /// caller that could get that order wrong would produce cycle
    /// counts for a pipeline that cannot run.
    ///
    /// Fetch advances a cycle at a time, delivering
    /// `fetch_bytes_per_cycle` when the FIFO has room and recording
    /// backpressure when it does not. Every fetch cycle is a stall,
    /// because the datapath has nothing to do during it.
    ///
    /// Issue takes `ceil(weights / weight_lanes)` cycles, and each
    /// issued weight is applied to the batch in
    /// `ceil(batch / batch_width)` cycles. The two multiply because a
    /// lane cannot take a new weight until it has finished the last
    /// one -- this model has no lane-level pipelining, which is
    /// exactly the fidelity saga 2 would add if a question needed it.
    pub fn process(&mut self, group: (usize, usize), batch: usize, config: &FabricConfig) {
        let (weights, bytes) = group;
        while self.fifo_level < bytes {
            if config.fifo_bytes - self.fifo_level < config.fetch_bytes_per_cycle {
                self.backpressure_cycles += 1;
            } else {
                self.fifo_level += config.fetch_bytes_per_cycle;
            }
            self.cycles += 1;
            self.stall_cycles += 1;
        }
        let issue = weights.div_ceil(config.weight_lanes);
        let apply = batch.div_ceil(config.batch_width);
        self.cycles += (issue * apply) as u64;
        self.fifo_level = self.fifo_level.saturating_sub(bytes);
    }

    /// Fraction of cycles the datapath was doing work.
    ///
    /// `None` for a run of zero cycles, where occupancy is undefined
    /// rather than zero.
    #[must_use]
    pub fn occupancy(&self) -> Option<f64> {
        (self.cycles > 0).then(|| {
            let busy = self.cycles - self.stall_cycles.min(self.cycles);
            widen(busy) / widen(self.cycles)
        })
    }
}

/// What one modelled scan produced.
///
/// Lives beside [`Pipeline`] because it is mostly a cycle result:
/// the accumulators come along because a model that did not compute
/// them could not be checked against anything.
#[derive(Debug)]
pub struct FabricOutcome {
    /// Accumulators. Bit-exact against `spm-gemv-ref`.
    pub bank: AccumulatorBank,
    /// Cycle behaviour of the modelled pipeline.
    pub pipeline: Pipeline,
    /// Weights consumed.
    pub weights: u64,
}

impl FabricOutcome {
    /// Cycles spent per weight consumed.
    ///
    /// The throughput figure this model reports, and deliberately not
    /// a duration: multiply by a clock period the day someone
    /// measures one.
    #[must_use]
    pub fn cycles_per_weight(&self) -> Option<f64> {
        (self.weights > 0).then(|| widen(self.pipeline.cycles) / widen(self.weights))
    }
}
