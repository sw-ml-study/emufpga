//! A parameter stream limited to a real device's bandwidth.
//!
//! Every measurement in docs/results.md before this read from a
//! page-cached file on a 64 GB machine, so the demanded-bandwidth
//! figures were **requirements a store would have to meet**, never
//! observations of one meeting them. This wraps any
//! [`spm_stream::WeightStream`] and makes it deliver at a chosen rate,
//! so the question "can a cheap sequential device feed this engine"
//! becomes a measurement.
//!
//! **Bandwidth only.** No seek latency, no queueing, no readahead, no
//! overlap between fetch and compute. That is deliberately consistent
//! with the rest of the stack rather than a new assumption:
//! `spm-stream-file` already refills synchronously, which is why
//! docs/results.md says `eta` measures a serial pipeline. A throttle
//! that modelled overlap would be modelling something the engine does
//! not do.
//!
//! What it adds that nothing else measures is **stall time**: how long
//! the engine spent waiting on the store rather than computing. That
//! is `eta` observed instead of derived.

mod stream;
mod throttle;

pub use throttle::Throttle;
