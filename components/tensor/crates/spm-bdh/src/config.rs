//! BDH's shape, from `BDHConfig` in the reference.

/// The reference defaults from `pathwaycom/bdh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BdhConfig {
    /// Levels the single parameter set is applied for.
    pub n_layer: usize,
    /// Model width, `D`.
    pub hidden: usize,
    /// Attention heads, `nh`.
    pub heads: usize,
    /// `mlp_internal_dim_multiplier`.
    pub multiplier: usize,
    /// Token count.
    pub vocab: usize,
}

impl Default for BdhConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            hidden: 256,
            heads: 4,
            multiplier: 128,
            vocab: 256,
        }
    }
}

impl BdhConfig {
    /// The sparse latent width per head, `N = multiplier * D / nh`.
    ///
    /// 8192 at the defaults -- thirty-two times the model width, which
    /// is why the latent rather than the weights dominates resident
    /// memory at any real sequence length.
    #[must_use]
    pub fn latent(&self) -> usize {
        self.multiplier * self.hidden / self.heads.max(1)
    }

    /// Streams in the rotating region: `encoder` and `encoder_v` per
    /// head, then the single `decoder`.
    #[must_use]
    pub fn rotating_streams(&self) -> usize {
        self.heads * 2 + 1
    }

    /// `(activation bytes, parameter bytes)` for `positions` tokens.
    ///
    /// Returned as a pair because the two are only ever meaningful
    /// against each other. This is the quantity that contradicts
    /// docs/plan.md section 3: the first grows linearly in sequence
    /// length, the second does not grow at all.
    ///
    /// The dominant activation term is `positions * heads * latent`,
    /// held **once**. The reference implementation holds `x_sparse`
    /// and `y_sparse` at the same time; folding the second `relu`
    /// into an in-place gate halves it.
    #[must_use]
    pub fn budget(&self, positions: usize) -> (usize, usize) {
        let (n, nh, d) = (self.latent(), self.heads, self.hidden);
        let live =
            positions * nh * n + 2 * positions * n + positions * nh * d + 2 * positions * d + n;
        let stored = 3 * nh * d * n + self.vocab * d;
        (live * size_of::<f32>(), stored * size_of::<f32>())
    }
}

/// `LayerNorm`'s epsilon, matching torch's default.
pub const EPS: f32 = 1e-5;

/// `get_freqs(theta=2**16)` in the reference.
pub const THETA: f32 = 65536.0;
