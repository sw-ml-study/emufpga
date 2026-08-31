//! One transformer block, reading its weights by subscript.
//!
//! A line-for-line counterpart to `spm_trm::Layer`. The activation
//! operators are literally the same functions from `spm_ops`; only the
//! four projections differ, and they differ only in where `W` came
//! from.

use crate::weights::ResidentWeights;
use spm_linear::LinearError;
use spm_ops::{multi_head, residual_norm, swiglu_batch};
use spm_trm::TrmConfig;

/// Scratch buffers for one block, allocated once and reused.
///
/// Identical to the streamed path's, and that is the point: the
/// activation side of the ledger does not change. What changes is the
/// weight side, which the streamed path never materialises.
pub struct ResidentLayer {
    qkv: Vec<f32>,
    attn: Vec<f32>,
    gate_up: Vec<f32>,
    folded: Vec<f32>,
    hidden: Vec<f32>,
}

impl ResidentLayer {
    /// Buffers for one block of `config`'s shape, `positions` long.
    #[must_use]
    pub fn new(config: &TrmConfig, positions: usize) -> Self {
        let (width, inter) = (config.hidden, config.intermediate());
        Self {
            qkv: vec![0.0; positions * width * 3],
            attn: vec![0.0; positions * width],
            gate_up: vec![0.0; positions * inter * 2],
            folded: vec![0.0; positions * inter],
            hidden: vec![0.0; positions * width],
        }
    }

    /// Runs block `layer` of the model over `state`.
    ///
    /// `base` is `layer * 4`: the four projections of a block sit
    /// consecutively in the consumption-order layout. The streamed
    /// path reaches them by arriving at them; this one computes their
    /// indices, which is the whole difference.
    ///
    /// # Errors
    /// Returns [`LinearError`] if a matrix disagrees with its shape.
    pub fn forward(
        &mut self,
        weights: &ResidentWeights,
        config: &TrmConfig,
        state: &mut [f32],
        layer: usize,
    ) -> Result<(), LinearError> {
        self.attention(weights, config, state, layer * 4)?;
        self.mlp(weights, config, state, layer * 4)
    }

    /// Attention sublayer: `qkv_proj`, heads, `o_proj`, residual, norm.
    fn attention(
        &mut self,
        weights: &ResidentWeights,
        config: &TrmConfig,
        state: &mut [f32],
        base: usize,
    ) -> Result<(), LinearError> {
        let (width, positions) = (config.hidden, state.len() / config.hidden);
        let shape = (config.heads, config.head_dim(), positions);
        weights.project(base, (width * 3, width), (state, positions), &mut self.qkv)?;
        multi_head(&self.qkv, shape, config.rope_base, &mut self.attn);
        let attended = (&*self.attn, positions);
        weights.project(base + 1, (width, width), attended, &mut self.hidden)?;
        residual_norm(state, &self.hidden, config.eps, width);
        Ok(())
    }

    /// `MLP` sublayer: `gate_up_proj`, `SwiGLU`, `down_proj`, residual.
    fn mlp(
        &mut self,
        weights: &ResidentWeights,
        config: &TrmConfig,
        state: &mut [f32],
        base: usize,
    ) -> Result<(), LinearError> {
        let (width, positions) = (config.hidden, state.len() / config.hidden);
        let inter = config.intermediate();
        weights.project(
            base + 2,
            (inter * 2, width),
            (state, positions),
            &mut self.gate_up,
        )?;
        swiglu_batch(&self.gate_up, &mut self.folded, inter, positions);
        let folded = (&*self.folded, positions);
        weights.project(base + 3, (width, inter), folded, &mut self.hidden)?;
        residual_norm(state, &self.hidden, config.eps, width);
        Ok(())
    }
}
