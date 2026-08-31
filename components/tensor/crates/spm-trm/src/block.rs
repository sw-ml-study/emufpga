//! One transformer block: attention then MLP, each post-normed.

use crate::config::TrmConfig;
use spm_linear::{LinearError, streamed};
use spm_ops::{multi_head, residual_norm, swiglu_batch};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Scratch buffers a block needs, allocated once and reused.
///
/// Reused across all 15 `L_level` calls and both layers, so a forward
/// pass allocates nothing in its hot path. Resident memory stays flat
/// while the weights rotate past -- which is the asymmetry the whole
/// architecture depends on.
pub struct Layer {
    qkv: Vec<f32>,
    attn: Vec<f32>,
    gate_up: Vec<f32>,
    folded: Vec<f32>,
    hidden: Vec<f32>,
}

impl Layer {
    /// Buffers for one block of `config`'s shape, `positions` long.
    #[must_use]
    pub fn new(config: &TrmConfig, positions: usize) -> Self {
        let width = config.hidden;
        let inter = config.intermediate();
        Self {
            qkv: vec![0.0; positions * width * 3],
            attn: vec![0.0; positions * width],
            gate_up: vec![0.0; positions * inter * 2],
            folded: vec![0.0; positions * inter],
            hidden: vec![0.0; positions * width],
        }
    }

    /// Runs one block over `state`, consuming four streams in order.
    ///
    /// `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` -- and there
    /// is no way to take them in any other order, because there is no
    /// seek. That is exactly the guarantee `layouts/*.order` provides,
    /// and the reason this reads as a straight line.
    ///
    /// Every projection is one sweep serving all positions at once: a
    /// weight is fetched once and applied `positions` times before
    /// being discarded, which is the reuse `Ps` measures.
    ///
    /// # Errors
    /// Returns [`LinearError`] if a stream ends early or disagrees
    /// with the expected shape.
    pub fn forward<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &TrmConfig,
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        self.attention(groups, config, state)?;
        self.mlp(groups, config, state)
    }

    /// Attention sublayer: `qkv_proj`, heads, `o_proj`, residual, norm.
    fn attention<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &TrmConfig,
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (width, positions) = (config.hidden, state.len() / config.hidden);
        streamed(
            groups,
            (width * 3, width),
            (state, positions),
            &mut self.qkv,
        )?;
        let shape = (config.heads, config.head_dim(), positions);
        multi_head(&self.qkv, shape, config.rope_base, &mut self.attn);
        streamed(
            groups,
            (width, width),
            (&self.attn, positions),
            &mut self.hidden,
        )?;
        residual_norm(state, &self.hidden, config.eps, width);
        Ok(())
    }

    /// `MLP` sublayer: `gate_up_proj`, `SwiGLU`, `down_proj`, residual.
    fn mlp<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &TrmConfig,
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (width, positions) = (config.hidden, state.len() / config.hidden);
        let inter = config.intermediate();
        streamed(
            groups,
            (inter * 2, width),
            (state, positions),
            &mut self.gate_up,
        )?;
        swiglu_batch(&self.gate_up, &mut self.folded, inter, positions);
        streamed(
            groups,
            (width, inter),
            (&self.folded, positions),
            &mut self.hidden,
        )?;
        residual_norm(state, &self.hidden, config.eps, width);
        Ok(())
    }
}
