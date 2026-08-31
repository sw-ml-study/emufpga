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
    /// `MLP` expansion factor. See [`TrmConfig::intermediate`] -- the
    /// intermediate width is NOT `hidden * expansion`.
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

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// `MLP` intermediate width, the way TRM actually computes it.
    ///
    /// `round(expansion * hidden * 2/3)`, aligned **up** to a multiple
    /// of 256. Not `hidden * expansion`, which is what this crate
    /// assumed until the reference implementation was read: for TRM
    /// that gives 2048 where the true value is 1536.
    ///
    /// The checkpoint settles it. `gate_up_proj` is `(3072, 512)` and
    /// 3072 is 2 x 1536, while `down_proj` is `(512, 1536)`. The
    /// earlier tests missed it because they generated their shapes
    /// from the same wrong formula they were checking -- self
    /// consistent, and self confirming.
    ///
    /// Integer arithmetic throughout: `(n + d/2) / d` rounds to
    /// nearest without a float ever appearing, so there is no
    /// platform-dependent rounding in a shape calculation.
    #[must_use]
    pub const fn intermediate(&self) -> usize {
        let numerator = self.expansion * self.hidden * 2;
        let rounded = (numerator + 1) / 3;
        rounded.div_ceil(256) * 256
    }
}
