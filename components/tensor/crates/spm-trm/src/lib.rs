//! TRM's recursion, driven off a rotating parameter stream.
//!
//! The first rung of the model ladder. 7M parameters, small enough
//! that every number is checkable by hand, before 27M (HRM), BDH,
//! SMOL and anything larger.
//!
//! # Why TRM suits this architecture unusually well
//!
//! A forward pass is `H_cycles * (L_cycles + 1)` = 15 `L_level` calls,
//! and every one sweeps the *same* eight matrices. At
//! `halt_max_steps` 16 that is up to 240 sweeps of the same 6,815,744
//! weights per puzzle.
//!
//! That is the rotating parameter store from docs/research.txt,
//! arriving for free because the model is recursive rather than deep.
//! A dense transformer touches each weight once per token; TRM
//! touches its whole weight set fifteen times per forward. The
//! architecture's central problem -- amortizing the cost of a scan --
//! is one this model solves for us.
//!
//! It also means `Ps` as currently defined **under-reports reuse by
//! 15x** for this model. Scan productivity counts applications per
//! weight *read*, and reads batch reuse only; recursion depth is a
//! second axis it cannot see. [`Forward::scan_productivity`] counts
//! both.
//!
//! # What is streamed and what is not
//!
//! The four projections per layer stream. Everything else -- the
//! residuals, `rms_norm`, `RoPE`, softmax, `SwiGLU`, and the two
//! latent states `z_H` and `z_L` -- stays resident, which docs/plan.md
//! section 3 allows explicitly. `z_H` and `z_L` are `seq_len x 512`
//! floats: kilobytes against megabytes of weights, and exactly the
//! asymmetry the architecture exploits.
//!
//! # What this does NOT establish
//!
//! That the model produces correct maze solutions. Verifying that
//! needs the published implementation, and torch is not installed.
//! What is established here is that the streamed path and a resident
//! path agree bit for bit on identical weights -- a statement about
//! mechanism, not about the model.

mod block;
mod config;
mod recursion;

pub use block::Layer;
pub use config::TrmConfig;
pub use recursion::{Forward, forward};
