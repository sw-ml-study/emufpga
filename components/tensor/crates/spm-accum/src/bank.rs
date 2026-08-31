//! A bank of accumulators per batch lane.

/// `lanes` independent banks of `rows` accumulators.
///
/// Stored **row-major**: the `lanes` accumulators for one output row
/// are adjacent, so applying a weight walks contiguous memory.
///
/// The obvious layout is the other one -- lane-major, so a lane's
/// outputs come back as a slice. Measured, it is much worse. A weight
/// touches every lane at one row, so lane-major strides by `rows`
/// between lanes: 4 KiB apart for a 1024-row matrix, one cache line
/// per lane, defeating the prefetcher. The step 006 sweep showed
/// useful throughput collapsing from 2113 to 434 million weights per
/// second going from batch 32 to batch 64, as the working set passed
/// L1. Row-major turns that inner loop into 64 contiguous floats.
///
/// It also matches the hardware. In the fabric the lanes for one row
/// are adjacent registers, not a stride away, so the reference now
/// has the same locality the RTL will.
///
/// The cost is that reading one lane's results gathers with a stride.
/// That happens once per operation; accumulation happens once per
/// nonzero weight per lane.
#[derive(Clone, Debug, PartialEq)]
pub struct AccumulatorBank {
    /// Batch lanes. Each holds an independent activation vector.
    pub lanes: usize,
    /// Accumulators per lane, one per output row.
    pub rows: usize,
    data: Vec<f32>,
}

impl AccumulatorBank {
    /// A zeroed bank.
    #[must_use]
    pub fn new(lanes: usize, rows: usize) -> Self {
        Self {
            lanes,
            rows,
            data: vec![0.0; lanes * rows],
        }
    }

    /// Applies one ternary weight to `row` across every lane.
    ///
    /// `activations` holds the already-scaled activation for each
    /// lane, so this performs an add or a subtract and **never a
    /// multiply**. In hardware `negative` is bit 1 of the weight code
    /// driving the add/subtract select, and the caller has already
    /// decided the weight is nonzero using bit 0.
    ///
    /// # Panics
    /// Panics if `row` is out of range or `activations` is shorter
    /// than `lanes`. Both are engine bugs, not input errors.
    pub fn accumulate(&mut self, row: usize, negative: bool, activations: &[f32]) {
        assert!(row < self.rows, "row {row} out of range");
        for (lane, activation) in activations.iter().take(self.lanes).enumerate() {
            let slot = &mut self.data[row * self.lanes + lane];
            if negative {
                *slot -= activation;
            } else {
                *slot += activation;
            }
        }
    }

    /// The accumulators for one lane, gathered.
    ///
    /// Allocates, because the storage layout is row-major (see the
    /// type docs). Called once per operation, against an inner loop
    /// that runs once per nonzero weight per lane.
    ///
    /// # Panics
    /// Panics if `lane` is out of range.
    #[must_use]
    pub fn lane(&self, lane: usize) -> Vec<f32> {
        assert!(lane < self.lanes, "lane {lane} out of range");
        (0..self.rows)
            .map(|row| self.data[row * self.lanes + lane])
            .collect()
    }

    /// Zeroes every accumulator, ready for the next operation.
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }
}
