//! SmolLM2-135M streamed. Rung 4, and the first non-recursive one.
//!
//! `HuggingFaceTB/SmolLM2-135M`, a Llama-shaped transformer: 30
//! distinct layers, 134,515,008 parameters, grouped-query attention,
//! tied embeddings.
//!
//! **This rung has no rotating region.** TRM, HRM and BDH each re-read
//! one small weight set many times per forward; here every weight is
//! read exactly once and there is no rewind inside a forward pass.
//! The free amortization the first three rungs enjoyed is gone.
//!
//! That is not a worse fit for a serial parameter store. It is the
//! best one yet: a forward pass is 210 streams read strictly in order,
//! start to finish, with no re-reads to pay for. What recursion bought
//! the earlier rungs was a small **working set**, not reuse --
//! arithmetic intensity is `batch / 4` MACs per weight-byte for any
//! f32 model read once, recursion or not. docs/results.md works this
//! out.
//!
//! The cost this rung pays instead is residency. `SmolLM` ties its
//! embeddings, and `embed_tokens` is 21% of the model. An embedding is
//! gathered by token id, so it cannot be swept to serve one token and
//! stays in RAM. Against 0.13% resident for TRM and 0.05% for HRM,
//! that is the honest price of a real vocabulary.

mod config;
mod layer;
mod model;

pub use config::SmolConfig;
pub use layer::Layer;
pub use model::{Resident, forward};
