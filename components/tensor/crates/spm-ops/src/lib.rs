//! The operators that run on resident activations.
//!
//! Everything here works on the **working set**, not the parameter
//! store: normalization, the `SwiGLU` nonlinearity, rotary embeddings,
//! softmax attention. docs/plan.md section 3 puts activations,
//! accumulators and small state in ordinary random-access memory on
//! purpose -- the restriction this project enforces is on weights
//! alone.
//!
//! That division is the whole design, so it is worth stating why
//! trying to extend it here would be a mistake. A softmax needs the
//! whole row before it can normalize; attention needs every key
//! before it can weight a value. Making those sequential would buy
//! nothing, because they touch kilobytes of activations rather than
//! gigabytes of parameters. The asymmetry between the two is the
//! reason the architecture works at all.
//!
//! None of these operators carries learned weights. TRM's `rms_norm`
//! has no gain vector -- which is why its checkpoint contains no norm
//! tensors -- and rotary embeddings are computed rather than stored.

mod attention;
mod nonlinear;
mod norm;

pub use attention::{attend, multi_head, rope};
pub use nonlinear::{silu, swiglu, swiglu_batch};
pub use norm::{residual_norm, rms_norm};
