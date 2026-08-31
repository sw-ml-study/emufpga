//! SmolLM2-135M's shape, from its published `config.json`.

/// The published configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmolConfig {
    /// `hidden_size`.
    pub hidden: usize,
    /// `intermediate_size`.
    pub intermediate: usize,
    /// `num_hidden_layers`.
    pub layers: usize,
    /// `num_attention_heads`.
    pub heads: usize,
    /// `num_key_value_heads`. Fewer than `heads`: this is GQA.
    pub kv_heads: usize,
    /// `rope_theta`. 100000 here, not the near-universal 10000.
    pub rope_base: f32,
    /// `rms_norm_eps`.
    pub eps: f32,
    /// `vocab_size`.
    pub vocab: usize,
}

impl Default for SmolConfig {
    fn default() -> Self {
        Self {
            hidden: 576,
            intermediate: 1536,
            layers: 30,
            heads: 9,
            kv_heads: 3,
            rope_base: 100_000.0,
            eps: 1e-5,
            vocab: 49152,
        }
    }
}

impl SmolConfig {
    /// Width of one attention head, 64 here.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads.max(1)
    }

    /// Combined width of the key or value projection, `kv_heads * head_dim`.
    ///
    /// 192, a third of the model width. GQA's whole point: keys and
    /// values are a third the size queries are.
    #[must_use]
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim()
    }

    /// Streams swept per forward: seven per layer, read once each.
    ///
    /// No rotating region, so this is also the total streamed count
    /// for a whole forward pass -- unlike every earlier rung, where it
    /// was the count per sweep and the sweep repeated.
    #[must_use]
    pub fn streams(&self) -> usize {
        self.layers * 7
    }
}
