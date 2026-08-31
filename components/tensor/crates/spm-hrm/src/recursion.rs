//! The two-module recursion, and the rewind pattern it needs.

use crate::config::HrmConfig;
use crate::report::HrmForward;
use spm_linear::LinearError;
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;
use spm_trm::Layer;

/// Runs one HRM forward pass over the rotating region.
///
/// `low` and `high` are the two modules' layer buffers. Both latent
/// states are the caller's and stay resident.
///
/// The rewind pattern is the whole point of this function, so it is
/// worth stating precisely: a rewind happens before every low sweep
/// **except the very first**, and never before a high sweep. The last
/// low sweep of an outer cycle leaves the cursor exactly where the
/// high module begins, so high continues forward with no seek. That
/// property depends entirely on low preceding high in the file: the
/// last low sweep of an outer cycle leaves the cursor exactly where
/// the high module begins, so the high sweep needs no seek.
///
/// # Errors
/// Returns [`LinearError`] if a sweep ends early or the stream fails.
pub fn forward<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &HrmConfig,
    states: (&mut [f32], &mut [f32]),
    modules: (&mut [Layer], &mut [Layer]),
    input: &[f32],
) -> Result<HrmForward, LinearError> {
    let ((z_low, z_high), (low, high)) = (states, modules);
    let seen = distinct(groups, config.rotating_streams());
    let mut report = HrmForward::new(z_low.len() / config.block.hidden, seen);
    for sweep_index in 0..config.h_cycles * config.l_cycles {
        if sweep_index > 0 {
            rewind(groups)?;
            report.rewinds += 1;
        }
        sweep(groups, &config.block, low, z_low, (z_high, Some(input)))?;
        report.low_sweeps += 1;
        if sweep_index % config.l_cycles == config.l_cycles - 1 {
            sweep(groups, &config.block, high, z_high, (z_low, None))?;
            report.high_sweeps += 1;
        }
    }
    report.weights_read = report.weights_distinct * config.h_cycles as u64;
    Ok(report)
}

/// Runs one module: inject, then every layer.
///
/// `ReasoningModule.forward` adds its injection to the hidden state
/// before the layers run, and the recursion supplies a different one
/// each time: the low module gets `z_high + input`, the high module
/// gets `z_low`. Omitting it produces a model that runs, stays finite
/// and is simply not HRM -- which is how it survived the first round
/// of tests here, since those checked sweep counts and finiteness.
fn sweep<S: WeightStream>(
    groups: &mut GroupStream<S>,
    block: &spm_trm::TrmConfig,
    layers: &mut [Layer],
    state: &mut [f32],
    inject: (&[f32], Option<&[f32]>),
) -> Result<(), LinearError> {
    for (index, slot) in state.iter_mut().enumerate() {
        *slot += inject.0[index] + inject.1.map_or(0.0, |extra| extra[index]);
    }
    for layer in layers.iter_mut() {
        layer.forward(groups, block, state)?;
    }
    Ok(())
}

/// Returns the stream to its first group, between operations.
fn rewind<S: WeightStream>(groups: &mut GroupStream<S>) -> Result<(), LinearError> {
    groups.rewind().map_err(|e| LinearError::Stream {
        detail: e.to_string(),
    })
}

/// Weights in the rotating region, from the stream directory.
fn distinct<S: WeightStream>(groups: &GroupStream<S>, streams: usize) -> u64 {
    groups
        .descriptors
        .iter()
        .take(streams)
        .map(|d| u64::from(d.rows) * u64::from(d.cols))
        .sum()
}
