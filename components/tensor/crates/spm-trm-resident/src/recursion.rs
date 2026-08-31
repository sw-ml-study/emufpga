//! The same recursion, with nothing to rewind.

use crate::block::ResidentLayer;
use crate::weights::ResidentWeights;
use spm_linear::LinearError;
use spm_trm::TrmConfig;

/// Runs `config.level_calls()` sweeps over resident weights.
///
/// The structural counterpart to `spm_trm::forward`, and the
/// difference is a single absence: there is no rewind. A rotating
/// parameter store re-reads its region once per call, so the streamed
/// path issues 14 rewinds per forward; here the weights never left, so
/// re-reading them is free and invisible.
///
/// That absence is the streamed path's cost and the resident path's
/// price. Which one is cheaper depends entirely on whether the
/// parameters fit, which is the question the ladder exists to answer.
///
/// # Errors
/// Returns [`LinearError`] if a matrix disagrees with its shape.
pub fn forward(
    weights: &ResidentWeights,
    config: &TrmConfig,
    state: &mut [f32],
    layers: &mut [ResidentLayer],
) -> Result<(), LinearError> {
    for _ in 0..config.level_calls() {
        for (index, layer) in layers.iter_mut().enumerate() {
            layer.forward(weights, config, state, index)?;
        }
    }
    Ok(())
}
