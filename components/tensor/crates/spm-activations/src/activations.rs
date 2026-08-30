//! One activation vector per batch lane.

/// One activation vector per batch lane, held in random-access
/// memory.
///
/// docs/plan.md section 3 allows this explicitly: activations,
/// accumulators and scales may use ordinary RAM. Only the parameter
/// stream is restricted. For a 70B model these are kilobytes against
/// tens of gigabytes, which is the whole asymmetry the architecture
/// exploits.
pub struct Activations {
    /// Batch lanes.
    pub lanes: usize,
    /// Activations per lane.
    pub cols: usize,
    values: Vec<f32>,
}

impl Activations {
    /// Holds `lanes` copies of `values`, one per batch lane.
    ///
    /// Every lane starting from the same vector is the configuration
    /// the batch-invariance test needs: with identical inputs, every
    /// lane must produce an identical result, or the batch dimension
    /// is wired wrong.
    #[must_use]
    pub fn broadcast(lanes: usize, values: &[f32]) -> Self {
        let mut all = Vec::with_capacity(lanes * values.len());
        for _ in 0..lanes {
            all.extend_from_slice(values);
        }
        Self {
            lanes,
            cols: values.len(),
            values: all,
        }
    }

    /// Holds a distinct activation vector per lane, laid out
    /// lane-major.
    #[must_use]
    pub fn per_lane(lanes: usize, cols: usize, values: Vec<f32>) -> Self {
        Self {
            lanes,
            cols,
            values,
        }
    }

    /// Writes `scale * x[col]` for every lane into `into`.
    ///
    /// **This is the only multiply in the engine.** It runs once per
    /// (group, column) pair, not once per weight. Everything the
    /// accumulators do afterwards is an add or a subtract.
    ///
    /// Writes into a caller-owned buffer rather than returning a
    /// slice, so the scan allocates nothing per column and holds no
    /// borrow across the inner loop. In the fabric this buffer is a
    /// small register file sized at synthesis, not something that
    /// grows.
    pub fn scale_column(&self, scale: f32, col: usize, into: &mut [f32]) {
        for (lane, slot) in into.iter_mut().take(self.lanes).enumerate() {
            *slot = scale * self.values[lane * self.cols + col];
        }
    }
}
