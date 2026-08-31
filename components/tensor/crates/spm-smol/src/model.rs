//! The 30-layer sweep, and the resident 21%.

use crate::config::SmolConfig;
use crate::layer::Layer;
use spm_linear::LinearError;
use spm_smol_ops::scaled_rms_norm;
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// What cannot be streamed, and why each entry is here.
///
/// `embed` is 21% of this model. An embedding is gathered by token id,
/// so a sweep would have to read all 49,152 rows to serve one token.
/// `SmolLM` ties its embeddings, so the same table is also the output
/// projection -- and since it is resident for the gather anyway,
/// streaming a transposed second copy would be pure waste.
///
/// The alternative is real and is not taken here: a sweep that
/// *selects* the rows it wants would keep residency at one group, at
/// the cost of reading 28 MB per forward. Which wins depends on
/// sequence length, and docs/results.md gives the arithmetic.
pub struct Resident<'a> {
    /// `(vocab, hidden)` row-major: row `t` is token `t`'s embedding.
    pub embed: &'a [f32],
    /// Per layer, the attention and MLP norm scales.
    pub norms: &'a [(Vec<f32>, Vec<f32>)],
    /// The final norm before the output projection.
    pub final_norm: &'a [f32],
}

/// Embeds tokens, sweeps every layer once, norms, and projects.
///
/// **No rewind.** The 210 weight matrices are read strictly in order,
/// start to finish, and the pass ends at the end of the stream. Every
/// earlier rung rewound; this one has nothing to rewind to.
///
/// # Errors
/// Returns [`LinearError`] if a stream ends early or disagrees with
/// its shape.
pub fn forward<S: WeightStream>(
    groups: &mut GroupStream<S>,
    config: &SmolConfig,
    resident: &Resident<'_>,
    layer: &mut Layer,
    io: (&[u32], &mut [f32], &mut [f32]),
) -> Result<(), LinearError> {
    let (tokens, state, logits) = io;
    let d = config.hidden;
    for (position, token) in tokens.iter().enumerate() {
        let row = *token as usize * d;
        state[position * d..(position + 1) * d].copy_from_slice(&resident.embed[row..row + d]);
    }
    for norms in resident.norms.iter().take(config.layers) {
        layer.forward(groups, config, (&norms.0, &norms.1), state)?;
    }
    scaled_rms_norm(state, resident.final_norm, config.eps, d);
    project(resident.embed, state, (config.vocab, d), logits);
    Ok(())
}

/// The tied output projection, from the resident embedding table.
///
/// `logits[p][v] = dot(state[p], embed[v])`. Resident rather than
/// streamed for the reason [`Resident`] gives.
fn project(embed: &[f32], state: &[f32], shape: (usize, usize), logits: &mut [f32]) {
    let (vocab, width) = shape;
    let positions = state.len() / width;
    for position in 0..positions {
        let row = &state[position * width..(position + 1) * width];
        for token in 0..vocab {
            let weights = &embed[token * width..(token + 1) * width];
            let dot: f32 = core::iter::zip(row, weights).map(|(a, b)| a * b).sum();
            logits[position * vocab + token] = dot;
        }
    }
}
