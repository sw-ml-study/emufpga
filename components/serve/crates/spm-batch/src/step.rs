//! One decode step: the model read once, every client served.

use crate::inner::{attention_half, mlp_half};
use crate::session::{Client, Scratch, StepReport};
use spm_linear::LinearError;
use spm_smol::{Resident, SmolConfig};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Runs one decode step for every client in `clients`.
///
/// Sweeps the weights **once**. Each client contributes the token it
/// is holding and receives one next-token logit vector, so the weight
/// bytes are divided by `clients.len()` tokens produced.
///
/// # Errors
/// Returns [`LinearError`] if a stream ends early or disagrees with
/// its shape.
pub fn decode_step<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &SmolConfig,
    resident: &Resident<'_>,
    work: (&mut [Client], &mut Scratch),
) -> Result<StepReport, LinearError> {
    let (clients, scratch) = work;
    let d = config.hidden;
    for (index, client) in clients.iter().enumerate() {
        let row = client.token as usize * d;
        scratch.states[index * d..(index + 1) * d].copy_from_slice(&resident.embed[row..row + d]);
    }
    for layer in 0..config.layers {
        one_layer(groups, config, resident, (clients, scratch), layer)?;
    }
    for client in clients.iter_mut() {
        client.cache.len += 1;
    }
    crate::inner::finish(config, resident, clients, scratch);
    Ok(StepReport::for_sweep(groups, config, clients.len()))
}

/// One layer, for every client, off one pass of its seven streams.
fn one_layer<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &SmolConfig,
    resident: &Resident<'_>,
    work: (&mut [Client], &mut Scratch),
    layer: usize,
) -> Result<(), LinearError> {
    let (clients, scratch) = work;
    let norms = &resident.norms[layer];
    attention_half(groups, config, (&norms.0, layer), (clients, scratch))?;
    mlp_half(groups, config, &norms.1, scratch)
}
