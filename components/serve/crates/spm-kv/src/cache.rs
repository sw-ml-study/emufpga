//! Keys and values for one client, one layer at a time.

/// One client's keys and values for every layer.
///
/// Laid out per layer as `context x kv_width`, so a layer's keys are
/// contiguous and the prefix a query attends over is a single slice.
///
/// Sized for `context` up front rather than grown: an allocation in
/// the middle of a decode step is the kind of jitter that makes a
/// timing measurement unreadable, and the bound is known.
pub struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    width: usize,
    context: usize,
    /// Positions written so far. Also this client's next position.
    pub len: usize,
}

impl KvCache {
    /// A cache for `layers` layers, `context` positions, `width` wide.
    #[must_use]
    pub fn new(layers: usize, context: usize, width: usize) -> Self {
        Self {
            keys: vec![0.0; layers * context * width],
            values: vec![0.0; layers * context * width],
            width,
            context,
            len: 0,
        }
    }

    /// Bytes this cache occupies. The bill for serving one client.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<f32>()
    }

    /// Appends one position's key and value for `layer`.
    ///
    /// Does not advance `len`: a decode step writes every layer at the
    /// same position, so the caller advances once at the end of the
    /// step. Advancing here would put layer 1 a position ahead of
    /// layer 0, which produces fluent-looking wrong output rather than
    /// an error.
    ///
    /// # Panics
    /// Panics if the context is full, which is a scheduling bug.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        assert!(
            self.len < self.context,
            "context {} exhausted",
            self.context
        );
        let at = (layer * self.context + self.len) * self.width;
        self.keys[at..at + self.width].copy_from_slice(key);
        self.values[at..at + self.width].copy_from_slice(value);
    }

    /// The keys and values written so far for `layer`, as
    /// `(len + 1) x width` slices including the position just
    /// appended.
    #[must_use]
    pub fn prefix(&self, layer: usize) -> (&[f32], &[f32]) {
        let start = layer * self.context * self.width;
        let end = start + (self.len + 1) * self.width;
        (&self.keys[start..end], &self.values[start..end])
    }
}
