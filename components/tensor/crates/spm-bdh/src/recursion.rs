//! The level loop, the rewind, and the one stream read at the end.

use crate::config::BdhConfig;
use crate::level::Level;
use spm_linear::{LinearError, streamed};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Runs `n_layer` levels and then the output projection.
///
/// Rewinds before every level except the first, so each begins at
/// stream zero. After the last level the cursor sits exactly where
/// `lm_head` begins -- the rotating region ends there -- so the logits
/// are produced by **reading on**, with no rewind and no seek. That is
/// the same structure HRM's `[low][high]` turned out to have, and it
/// is the reason `layouts/bdh.order` puts `lm_head` last.
///
/// Returns the rewinds issued, so a test can assert the store really
/// rotated rather than trusting that it did.
///
/// # Errors
/// Returns [`LinearError`] if a sweep ends early or the stream fails.
pub fn forward<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &BdhConfig,
    state: &mut [f32],
    level: &mut Level,
    logits: &mut [f32],
) -> Result<usize, LinearError> {
    let positions = state.len() / config.hidden;
    let mut rewinds = 0;
    for pass in 0..config.n_layer {
        if pass > 0 {
            rewind(groups)?;
            rewinds += 1;
        }
        level.forward(groups, config, state)?;
    }
    streamed(
        groups,
        (config.vocab, config.hidden),
        (state, positions),
        logits,
    )?;
    Ok(rewinds)
}

/// Returns the stream to its first group, between levels only.
fn rewind<S: WeightStream>(groups: &mut GroupStream<S>) -> Result<(), LinearError> {
    groups.rewind().map_err(|e| LinearError::Stream {
        detail: e.to_string(),
    })
}
