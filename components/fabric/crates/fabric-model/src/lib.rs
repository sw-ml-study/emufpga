//! A conceptual model of something FPGA-like.
//!
//! Not an FPGA simulator. There are no LUT4s here, no device profiles,
//! and nothing that predicts whether a design fits a part. What there
//! is: a pipeline with abstract knobs, executing a real `.spm` file and
//! reporting where it stalls.
//!
//! # Cycles are a unit, not a duration
//!
//! Nothing in this crate converts cycles to seconds, and no function
//! takes an fmax. No fabric clock has been measured (docs/plan.md
//! section 6 records fmax as Unknown for every board), and accepting
//! one would turn a conceptual model into a fidelity claim by the back
//! door. Throughput questions are answered in **cycles per weight**,
//! which becomes wall clock the day someone measures a clock and not
//! before.
//!
//! # The pipeline
//!
//! ```text
//!   parameter stream
//!         |
//!         v
//!   fetch  (fetch_bytes_per_cycle, after fetch_latency_cycles)
//!         |
//!         v
//!    FIFO  (fifo_bytes)
//!         |
//!         v
//!   issue  (weight_lanes weights per cycle)
//!         |
//!         v
//!  accumulate  (batch_width lanes per weight per cycle)
//! ```
//!
//! The interesting output is which side is waiting. A datapath that
//! starves on an empty FIFO and a FIFO that backs up against a slow
//! datapath are the two regimes docs/results.md's `eta` distinguishes,
//! and the two models should agree in direction.
//!
//! # Correctness is not approximate
//!
//! The cycle counts are a model; the arithmetic is not. This engine
//! must produce accumulators **bit-exact** against `spm-gemv-ref`.
//! That holds even with many lanes because the stream is column-major:
//! `weight_lanes` consecutive weights land on `weight_lanes` different
//! accumulators, so no accumulator sees a reordered summation. If
//! agreement is ever inexact, that is a finding to chase rather than a
//! tolerance to widen.

mod config;
mod cycles;
mod run;

pub use config::{FabricConfig, FabricError};
pub use cycles::{FabricOutcome, Pipeline};
pub use run::run_fabric;
