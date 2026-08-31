//! BDH's parameterless math: sparse latents and linear attention.
//!
//! Separate from `spm-ops` because none of it is shared. BDH's
//! attention carries no learned weights at all -- its `RoPE`
//! frequencies are a computed buffer -- and it rotates over the
//! **sparse latent** dimension `N` rather than over a head dimension
//! of the model width, which is a different operator wearing a
//! familiar name.
//!
//! Read from `pathwaycom/bdh`, not inferred. Two details here would be
//! silently wrong if guessed, and both are the sort postmortem 1 is
//! about:
//!
//! 1. `get_freqs` quantizes the index in **pairs** before the
//!    exponent, so adjacent even/odd entries share a frequency.
//! 2. The score matrix is `.tril(diagonal=-1)` -- strictly lower
//!    triangular, EXCLUDING the diagonal, so a position never attends
//!    to itself. A conventional causal mask includes it.

mod scores;
mod sparse;

pub use scores::{attend_heads, freqs, rotate, scores_times_values};
pub use sparse::{relu_into, scale_product_into};
