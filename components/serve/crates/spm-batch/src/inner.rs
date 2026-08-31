//! The halves of a layer, and what happens after the last one.

use crate::session::{Client, Scratch};
use spm_kv::{attend_cached, rotate_at};
use spm_linear::{LinearError, streamed};
use spm_ops::silu;
use spm_smol::{Resident, SmolConfig};
use spm_smol_ops::{add_into, pre_norm, scaled_rms_norm};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Pre-norm attention: q, k, v, per-client attention, o, residual.
///
/// # Errors
/// Returns [`LinearError`] if a stream ends early.
pub fn attention_half<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &SmolConfig,
    at: (&[f32], usize),
    work: (&mut [Client], &mut Scratch),
) -> Result<(), LinearError> {
    let (norm, layer) = at;
    let (clients, scratch) = work;
    let (d, kv, n) = (config.hidden, config.kv_width(), clients.len());
    pre_norm(&scratch.states, &mut scratch.normed, norm, config.eps, d);
    let batch = (&scratch.normed[..], n);
    streamed(groups, (d, d), batch, &mut scratch.q)?;
    streamed(groups, (kv, d), batch, &mut scratch.k)?;
    streamed(groups, (kv, d), batch, &mut scratch.v)?;
    attention(config, clients, scratch, layer);
    streamed(groups, (d, d), (&scratch.attn, n), &mut scratch.projected)?;
    add_into(&mut scratch.states, &scratch.projected);
    Ok(())
}

/// Pre-norm `SwiGLU` MLP: gate, up, silu gate, down, residual.
///
/// # Errors
/// Returns [`LinearError`] if a stream ends early.
pub fn mlp_half<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &SmolConfig,
    norm: &[f32],
    scratch: &mut Scratch,
) -> Result<(), LinearError> {
    let (d, i) = (config.hidden, config.intermediate);
    let n = scratch.states.len() / d;
    pre_norm(&scratch.states, &mut scratch.normed, norm, config.eps, d);
    let batch = (&scratch.normed[..], n);
    streamed(groups, (i, d), batch, &mut scratch.gate)?;
    streamed(groups, (i, d), batch, &mut scratch.up)?;
    for (slot, other) in scratch.gate.iter_mut().zip(&scratch.up) {
        *slot = silu(*slot) * other;
    }
    streamed(groups, (d, i), (&scratch.gate, n), &mut scratch.projected)?;
    add_into(&mut scratch.states, &scratch.projected);
    Ok(())
}

/// Per-client attention. The half that does not amortize.
///
/// Each client rotates at **its own** position and attends over **its
/// own** prefix, so nothing here is shared. It carries no weights,
/// which is exactly why it can be resident while the parameters
/// stream past.
pub(crate) fn attention(
    config: &SmolConfig,
    clients: &mut [Client],
    scratch: &mut Scratch,
    layer: usize,
) {
    let (d, kv, hd) = (config.hidden, config.kv_width(), config.head_dim());
    let (qh, kh, base) = ((config.heads, hd), (config.kv_heads, hd), config.rope_base);
    for (index, client) in clients.iter_mut().enumerate() {
        let at = client.cache.len;
        let (q, k) = (index * d..(index + 1) * d, index * kv..(index + 1) * kv);
        rotate_at(&mut scratch.q[q.clone()], qh, at, base);
        rotate_at(&mut scratch.k[k.clone()], kh, at, base);
        client
            .cache
            .append(layer, &scratch.k[k.clone()], &scratch.v[k]);
        let shape = (at + 1, config.heads, config.kv_heads, hd);
        let cached = client.cache.prefix(layer);
        attend_cached(&scratch.q[q.clone()], cached, shape, &mut scratch.attn[q]);
    }
}

/// Final norm and the tied output projection, per client.
pub(crate) fn finish(
    config: &SmolConfig,
    resident: &Resident<'_>,
    clients: &[Client],
    scratch: &mut Scratch,
) {
    let (d, vocab) = (config.hidden, config.vocab);
    scaled_rms_norm(&mut scratch.states, resident.final_norm, config.eps, d);
    for index in 0..clients.len() {
        let state = &scratch.states[index * d..(index + 1) * d];
        for token in 0..vocab {
            let row = &resident.embed[token * d..(token + 1) * d];
            let dot: f32 = core::iter::zip(state, row).map(|(a, b)| a * b).sum();
            scratch.logits[index * vocab + token] = dot;
        }
    }
}
