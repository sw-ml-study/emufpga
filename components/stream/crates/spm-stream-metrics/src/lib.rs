//! The numbers that decide whether the architecture works.
//!
//! Raw GB/s does not express what a Serial Parameter Machine is trying
//! to improve, so docs/plan.md section 4 defines five more. This crate
//! implements them over one struct of raw counters, [`ScanMetrics`],
//! so every benchmark in the project reports the same quantities the
//! same way.
//!
//! | Metric | Meaning |
//! |--------|---------|
//! | [`ScanMetrics::raw_bandwidth`] | bytes/s the backing store delivered |
//! | [`ScanMetrics::decoded_weights_per_sec`] | weights unpacked per second |
//! | [`ScanMetrics::useful_weights_per_sec`] | parameter applications per second |
//! | [`ScanMetrics::eta`] | can compute keep up with storage? |
//! | [`ScanMetrics::scan_productivity`] | `Ps` -- applications per weight read |
//! | [`ScanMetrics::residency`] | `Rp` -- parameter bytes in RAM over model size |
//!
//! `Ps` and `Rp` carry the whole economic argument. `Ps` is why
//! batching, `MoE` scheduling and speculative decoding matter: they
//! raise the useful work extracted from each byte that crosses the
//! bus. `Rp` is the claim that the model does not need to be resident;
//! conventional inference sits at `Rp ~= 1` and the goal is `Rp -> 0`
//! while activation memory stays nonzero.
//!
//! Every ratio returns `Option`, and `None` means the denominator was
//! zero. Returning `0.0` there would be a lie: a scan that decoded no
//! weights has no scan productivity, which is not the same as having
//! a scan productivity of zero.

mod model;
mod rates;
mod ratios;

pub use model::ScanMetrics;
