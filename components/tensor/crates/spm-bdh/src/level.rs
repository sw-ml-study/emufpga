//! One level: encode, attend, gate, project. Nine streams, in order.

use crate::config::{BdhConfig, EPS, THETA};
use spm_bdh_ops::{attend_heads, freqs, relu_into, scale_product_into};
use spm_linear::{LinearError, streamed};
use spm_ops::{layer_norm, residual_layer_norm};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Scratch for one level, allocated once and reused across all of
/// them.
///
/// `sparse` is the expensive one: `positions x (heads * latent)`, and
/// it serves as both `x_sparse` and `xy_sparse` because the gate
/// multiplies in place. Keeping one buffer rather than two halves the
/// largest allocation in the engine -- which at this model's shape is
/// the difference between 2 MB and 4 MB at 16 positions, and between
/// 67 MB and 134 MB at 256.
pub struct Level {
    sparse: Vec<f32>,
    rotated: Vec<f32>,
    latent: Vec<f32>,
    ykv: Vec<f32>,
    projected: Vec<f32>,
    turns: Vec<f32>,
}

impl Level {
    /// Buffers for `positions` tokens of `config`'s shape.
    #[must_use]
    pub fn new(config: &BdhConfig, positions: usize) -> Self {
        let (n, nh, d) = (config.latent(), config.heads, config.hidden);
        Self {
            sparse: vec![0.0; positions * nh * n],
            rotated: vec![0.0; positions * n],
            latent: vec![0.0; positions * n],
            ykv: vec![0.0; positions * nh * d],
            projected: vec![0.0; positions * d],
            turns: freqs(n, THETA),
        }
    }

    /// Runs one level over `state`, consuming nine streams in order.
    ///
    /// `encoder` per head, then `encoder_v` per head, then `decoder`.
    /// There is no way to take them in another order, because there is
    /// no seek -- which is what `layouts/bdh.order` exists to line up.
    ///
    /// # Errors
    /// Returns [`LinearError`] if a stream ends early or disagrees
    /// with the expected shape.
    pub fn forward<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        config: &BdhConfig,
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (n, nh, d) = (config.latent(), config.heads, config.hidden);
        self.encode(groups, (n, nh, d), state)?;
        self.gate(groups, (n, nh, d), state)
    }

    /// Sweeps `encoder` for every head into the sparse latent.
    ///
    /// `streamed` writes a compact `positions x latent` block, which
    /// is then scattered into the positions-major `sparse` buffer the
    /// decoder sweep needs. The scatter is the price of giving the
    /// final matmul a contiguous activation row.
    fn encode<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        shape: (usize, usize, usize),
        state: &[f32],
    ) -> Result<(), LinearError> {
        let (n, nh, d) = shape;
        let positions = state.len() / d;
        for head in 0..nh {
            streamed(groups, (n, d), (state, positions), &mut self.latent)?;
            for position in 0..positions {
                let at = position * nh * n + head * n;
                let src = &self.latent[position * n..(position + 1) * n];
                relu_into(src, &mut self.sparse[at..at + n]);
            }
        }
        let buffers = (&mut self.rotated[..], &mut self.ykv[..]);
        let turns = &self.turns;
        attend_heads(&self.sparse, state, (positions, n, d, nh), turns, buffers);
        layer_norm(&mut self.ykv, EPS, d);
        Ok(())
    }

    /// Sweeps `encoder_v` per head, folding `relu` into the gate.
    ///
    /// `sparse` holds `x_sparse` on entry and `xy_sparse` on exit: the
    /// product is applied in place, so no second buffer of this size
    /// ever exists.
    fn gate<S: WeightStream>(
        &mut self,
        groups: &mut GroupStream<S>,
        shape: (usize, usize, usize),
        state: &mut [f32],
    ) -> Result<(), LinearError> {
        let (n, nh, d) = shape;
        let positions = state.len() / d;
        for head in 0..nh {
            let span = positions * d;
            let values = &self.ykv[head * span..(head + 1) * span];
            streamed(groups, (n, d), (values, positions), &mut self.latent)?;
            for position in 0..positions {
                let at = position * nh * n + head * n;
                let src = &self.latent[position * n..(position + 1) * n];
                scale_product_into(src, &mut self.sparse[at..at + n]);
            }
        }
        let batch = (&self.sparse[..], positions);
        streamed(groups, (d, nh * n), batch, &mut self.projected)?;
        layer_norm(&mut self.projected, EPS, d);
        residual_layer_norm(state, &self.projected, EPS, d);
        Ok(())
    }
}
