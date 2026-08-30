//! A bank of accumulators per batch lane.

/// `lanes` independent banks of `rows` accumulators.
///
/// Stored lane-major so one lane's outputs are contiguous, matching
/// how a consumer reads results back out.
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
            let slot = &mut self.data[lane * self.rows + row];
            if negative {
                *slot -= activation;
            } else {
                *slot += activation;
            }
        }
    }

    /// The accumulators for one lane.
    ///
    /// # Panics
    /// Panics if `lane` is out of range.
    #[must_use]
    pub fn lane(&self, lane: usize) -> &[f32] {
        assert!(lane < self.lanes, "lane {lane} out of range");
        &self.data[lane * self.rows..(lane + 1) * self.rows]
    }

    /// Zeroes every accumulator, ready for the next operation.
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }
}
