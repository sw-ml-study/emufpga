//! HRM's two-module recursion over a rotating parameter stream.
//!
//! Rung 2 of the model ladder. 27,276,802 parameters, from
//! `zbloss/HRM-sudoku-extreme`.
//!
//! # What is new here, and what is not
//!
//! The *block* is not new. HRM's own source describes it as
//! self-attention, RMS norm, fully connected, RMS norm, with
//! post-norm residuals -- the same block TRM uses, which is
//! unsurprising since TRM was derived from HRM. Its config states the
//! same `hidden_size`, `num_attention_heads`, `expansion`,
//! `intermediate_size`, `rope_theta` and `rms_norm_eps`. So
//! [`spm_trm::Layer`] is reused rather than reimplemented, and it is
//! already verified numerically against a published implementation.
//!
//! What IS new is the recursion, and it is genuinely different:
//!
//! ```text
//! for h in 0..H_cycles:          // 2
//!   for l in 0..L_cycles:        // 2
//!     z_L = low_level(z_L, z_H + input)
//!   z_H = high_level(z_H, z_L)
//! ```
//!
//! Two distinct modules of four layers each, where TRM has one shared
//! module of two. That was the open risk when this step began: if the
//! rotating region needed to be two separate runs, `rewind` alone
//! could not serve it, because rewind returns to the start of the
//! stream and there is no seek to offer it an offset.
//!
//! # It does not need two regions
//!
//! With `[low][high]` contiguous and low first, a plain rewind-to-zero
//! serves the whole recursion. The last low sweep of an outer cycle
//! leaves the cursor exactly where the high-level module begins, so
//! the high sweep simply continues forward:
//!
//! ```text
//! sweep low            cursor -> first high stream
//! rewind, sweep low    cursor -> first high stream
//! sweep high           cursor -> end
//! rewind               cursor -> first low stream
//! ```
//!
//! That only works because low precedes high in the file. The
//! checkpoint's own ordering is alphabetical, which puts high first
//! and would require a backwards seek -- the same trap TRM's layout
//! had, in a second place.

mod config;
mod recursion;
mod report;

pub use config::HrmConfig;
pub use recursion::forward;
pub use report::HrmForward;
