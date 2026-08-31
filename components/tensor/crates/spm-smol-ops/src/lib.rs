//! What a Llama-shaped transformer needs and the earlier rungs did not.
//!
//! Three differences from TRM and HRM, each of which produces
//! plausible finite garbage if carried over by reflex:
//!
//! 1. **Grouped-query attention.** 9 query heads share 3 KV heads.
//!    `spm_ops::multi_head` is plain MHA.
//! 2. **Causal masking.** TRM and HRM see a whole puzzle at once and
//!    are unmasked; `SmolLM` is autoregressive. The mask **includes**
//!    the diagonal, unlike BDH's `.tril(diagonal=-1)`.
//! 3. **A learned norm scale.** Llama's RMS norm multiplies by a
//!    weight vector; TRM's has none.
//!
//! `spm_ops::rope` is reused unchanged. Its doc says "adjacent pair"
//! but the code rotates halves -- `head[i]` against `head[i + d/2]` --
//! which is the non-interleaved convention `SmolLM`'s config asks for
//! (`rope_interleaved: false`). Checked rather than assumed.

mod attention;
mod norm;

pub use attention::{grouped_causal, rotate_heads};
pub use norm::{add_into, pre_norm, scaled_rms_norm};
