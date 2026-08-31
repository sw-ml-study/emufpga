//! BDH, the Dragon Hatchling, streamed. Rung 3 of the model ladder.
//!
//! `pathwaycom/bdh`, arXiv 2509.26507. Read from the reference forward
//! pass rather than inferred from tensor names, per
//! docs/postmortem-1.md.
//!
//! **BDH is a rotating parameter store by construction.** Its loop
//! body carries no layer index: one parameter set is applied `n_layer`
//! times, exactly the property that made TRM a good first rung, and it
//! arrives here for free rather than by design. `lm_head` is read once
//! after the last level, so it sits after the rotating region and is
//! reached by reading on -- never by seeking back.
//!
//! **This rung contradicts an assumption the plan rests on.**
//! docs/plan.md section 3 justifies resident activations because they
//! are kilobytes while weights are megabytes. BDH's sparse latent is
//! `heads * positions * latent` floats, which for the reference
//! configuration is 2 MB at 16 positions and 67 MB at 256 -- against a
//! 101 MB weight set. The asymmetry the architecture depends on is
//! weaker here than at any earlier rung, and at long sequences it
//! inverts. docs/results.md carries the measurement.

mod config;
mod level;
mod recursion;

pub use config::BdhConfig;
pub use level::Level;
pub use recursion::forward;
