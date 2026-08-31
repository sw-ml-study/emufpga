//! TRM's shape, from its own `all_config.yaml`.

/// The configuration a forward pass needs.
///
/// Defaults are `yagizdevre/trm-maze-30x30`'s published values. They
/// are data rather than constants because HRM, the next rung, has the
/// same block shape with different numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrmConfig {
    /// Model width.
    pub hidden: usize,
    /// Attention heads. `hidden / heads` is the head dimension.
    pub heads: usize,
    /// `MLP` intermediate width multiplier.
    pub expansion: usize,
    /// Transformer blocks inside one `L_level` call.
    pub layers: usize,
    /// Outer recursion cycles.
    pub h_cycles: usize,
    /// Inner recursion cycles per outer cycle.
    pub l_cycles: usize,
    /// `RMSNorm` epsilon.
    pub eps: f32,
    /// `RoPE` base. An assumption until the reference confirms it.
    pub rope_base: f32,
}

impl Default for TrmConfig {
    fn default() -> Self {
        Self {
            hidden: 512,
            heads: 8,
            expansion: 4,
            layers: 2,
            h_cycles: 3,
            l_cycles: 4,
            eps: 1e-5,
            rope_base: 10_000.0,
        }
    }
}

impl TrmConfig {
    /// `L_level` calls per forward: `h_cycles * (l_cycles + 1)`.
    ///
    /// Each outer cycle runs `l_cycles` updates of `z_L` and then one
    /// of `z_H`, so the count is not `h * l`.
    #[must_use]
    pub const fn level_calls(&self) -> usize {
        self.h_cycles * (self.l_cycles + 1)
    }

    /// Streams swept per `L_level` call: four projections per layer.
    #[must_use]
    pub const fn streams_per_call(&self) -> usize {
        self.layers * 4
    }

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}
