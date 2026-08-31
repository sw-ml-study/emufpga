//! What a sweep produces.

use spm_stream_metrics::ScanMetrics;
use std::time::{Duration, Instant};

/// Which parameter store the scan read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The whole file already in RAM. An upper bound on storage
    /// bandwidth, and so a lower bound on `eta`.
    Memory,
    /// Read through a file, buffer by buffer. Closer to the real
    /// thing, though still without overlapped IO.
    File,
}

impl Backend {
    /// Short label for reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::File => "file",
        }
    }
}

/// One measured point: a backend at a batch size.
#[derive(Clone, Debug)]
pub struct BenchRow {
    /// Which store the weights came from.
    pub backend: Backend,
    /// Batch lanes.
    pub batch: usize,
    /// Metrics from the fastest of the repeated passes.
    ///
    /// Fastest rather than mean: a slow pass on a shared laptop
    /// measures the scheduler, not the engine. The spread is reported
    /// separately so a reader can see how much that choice mattered.
    pub best: ScanMetrics,
    /// Wall clock of the fastest pass.
    pub fastest: Duration,
    /// Wall clock of the slowest pass.
    pub slowest: Duration,
}

impl BenchRow {
    /// Spread between slowest and fastest, as a fraction of fastest.
    ///
    /// `None` if the fastest pass took no measurable time at all,
    /// which itself means the workload is too small to time.
    #[must_use]
    pub fn spread(&self) -> Option<f64> {
        let fastest = self.fastest.as_secs_f64();
        (fastest > 0.0).then(|| (self.slowest.as_secs_f64() - fastest) / fastest)
    }
}

/// A full sweep over batch sizes and backends.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// Every measured point, in the order they were taken.
    pub rows: Vec<BenchRow>,
    /// Measured cost of one timestamp pair on this machine.
    pub timer_overhead: Duration,
    /// Passes per point.
    pub repeat: usize,
}

/// Where the sweep sits relative to `eta == 1`.
///
/// Three outcomes, kept distinct because conflating them is how a
/// benchmark starts lying. "Compute-bound before the sweep began" is
/// not the same claim as "compute became the limit at batch 8", and
/// reporting the first as though it were the second would invent a
/// measurement that was never made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossover {
    /// `eta` was already below 1 at the smallest batch swept. The
    /// engine is compute-bound everywhere in range and the crossing
    /// point lies below it, unmeasured.
    AlreadyBelow {
        /// Smallest batch size swept.
        smallest: usize,
    },
    /// `eta` fell below 1 at this batch size, having been at or above
    /// 1 at the batch before it.
    At {
        /// The batch size where the crossing was observed.
        batch: usize,
    },
    /// `eta` stayed at or above 1 throughout. Storage remained the
    /// limit across the whole sweep.
    NotReached,
}

impl Sweep {
    /// Where `backend` sits relative to `eta == 1`.
    #[must_use]
    pub fn crossover(&self, backend: Backend) -> Crossover {
        let mut rows = self
            .rows
            .iter()
            .filter(|row| row.backend == backend)
            .peekable();
        let Some(first) = rows.peek().copied() else {
            return Crossover::NotReached;
        };
        if first.best.eta().is_some_and(|eta| eta < 1.0) {
            return Crossover::AlreadyBelow {
                smallest: first.batch,
            };
        }
        rows.find(|row| row.best.eta().is_some_and(|eta| eta < 1.0))
            .map_or(Crossover::NotReached, |row| Crossover::At {
                batch: row.batch,
            })
    }
}

/// Measures the cost of one timestamp pair on this machine.
///
/// Lives beside [`Sweep`] because it is the only producer of that
/// struct's `timer_overhead` field. The engine timestamps every scale
/// group twice; if that cost is comparable to the work between the
/// timestamps, the reported split between storage and compute is
/// noise. Measuring it lets a report say so rather than leaving a
/// reader to assume otherwise.
#[must_use]
pub fn timer_overhead() -> Duration {
    const SAMPLES: u32 = 10_000;
    let started = Instant::now();
    for _ in 0..SAMPLES {
        let inner = Instant::now();
        std::hint::black_box(inner.elapsed());
    }
    started.elapsed() / SAMPLES
}
