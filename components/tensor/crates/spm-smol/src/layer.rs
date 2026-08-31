//! One Llama block: pre-norm attention, pre-norm MLP, seven streams.

use crate::config::SmolConfig;
use spm_linear::{LinearError, streamed};
use spm_ops::silu;
use spm_smol_ops::{add_into, grouped_causal, pre_norm, rotate_heads};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Scratch for one block, allocated once and reused across all 30.
pub struct Layer {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    projected: Vec<f32>,
}

impl Layer {
    /// Buffers for `positions` tokens of `config`'s shape.
    #[must_use]
    pub fn new(config: &SmolConfig, positions: usize) -> Self {
        let (d, i, kv) = (config.hidden, config.intermediate, config.kv_width());
        Self {
            normed: vec![0.0; positions * d],
            q: vec![0.0; positions * d],
            k: vec![0.0; positions * kv],
            v: vec![0.0; positions * kv],
            attn: vec![0.0; positions * d],
            gate: vec![0.0; positions * i],
            up: vec![0.0; positions * i],
            projected: vec![0.0; positions * d],
        }
    }

    /// Runs one block, consuming q, k, v, o, gate, up, down in order.
    ///
    /// `norms` is the pair of RMS scales for this layer, which are
    /// resident: they are 576 floats each and gathered, not swept.
    ///
    /// # Errors
    /// Returns [`LinearError`] if a stream ends early or disagrees
    /// with its shape.
    pub fn forward<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &SmolConfig,
        norms: (&[f32], &[f32]),
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        self.attention(groups, config, norms.0, state)?;
        self.mlp(groups, config, norms.1, state)
    }

    /// Pre-norm attention: norm, q/k/v, `RoPE`, GQA, o, residual.
    ///
    /// **Pre-norm, not post-norm.** Llama norms the input to the
    /// sublayer and adds the raw residual; TRM and HRM norm the sum.
    /// Swapping them leaves the output finite and wrong.
    fn attention<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &SmolConfig,
        norm: &[f32],
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (d, kv, hd) = (config.hidden, config.kv_width(), config.head_dim());
        let positions = state.len() / d;
        pre_norm(state, &mut self.normed, norm, config.eps, d);
        let batch = (&self.normed[..], positions);
        streamed(groups, (d, d), batch, &mut self.q)?;
        streamed(groups, (kv, d), batch, &mut self.k)?;
        streamed(groups, (kv, d), batch, &mut self.v)?;
        let (heads, kv_heads) = (config.heads, config.kv_heads);
        rotate_heads(&mut self.q, (positions, heads, hd), config.rope_base);
        rotate_heads(&mut self.k, (positions, kv_heads, hd), config.rope_base);
        let shape = (positions, heads, kv_heads, hd);
        grouped_causal(&self.q, (&self.k, &self.v), shape, &mut self.attn);
        streamed(groups, (d, d), (&self.attn, positions), &mut self.projected)?;
        add_into(state, &self.projected);
        Ok(())
    }

    /// Pre-norm `SwiGLU` MLP: norm, gate and up, silu gate, down, residual.
    ///
    /// Llama ships `gate_proj` and `up_proj` as two tensors where TRM
    /// fuses them into one. Two streams, and the gate is the one that
    /// gets `silu` -- checked against the reference, not assumed.
    fn mlp<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &SmolConfig,
        norm: &[f32],
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (d, i) = (config.hidden, config.intermediate);
        let positions = state.len() / d;
        pre_norm(state, &mut self.normed, norm, config.eps, d);
        let batch = (&self.normed[..], positions);
        streamed(groups, (i, d), batch, &mut self.gate)?;
        streamed(groups, (i, d), batch, &mut self.up)?;
        for (slot, other) in self.gate.iter_mut().zip(&self.up) {
            *slot = silu(*slot) * other;
        }
        streamed(groups, (d, i), (&self.gate, positions), &mut self.projected)?;
        add_into(state, &self.projected);
        Ok(())
    }
}
