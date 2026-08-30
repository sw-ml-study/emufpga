//! Operation descriptors and consumption-order tiling.
//!
//! One descriptor per stream, fixed 32 bytes, all integers
//! little-endian:
//!
//! | offset | size | field |
//! |--------|------|-------|
//! | 0      | 4    | `rows` (M, outputs) |
//! | 4      | 4    | `cols` (N, inputs) |
//! | 8      | 4    | `group_size` (G, weights per scale group) |
//! | 12     | 1    | `encoding` profile |
//! | 13     | 1    | reserved, zero |
//! | 14     | 2    | `lane_count` |
//! | 16     | 16   | reserved, zero |
//!
//! # Consumption order
//!
//! For `y = Wx`, weights are stored **column-major**: stream position
//! `k` holds `W[k % rows][k / rows]`. Consecutive positions walk down
//! a column, so the engine holds one activation `x[j]` while an entire
//! column streams past, accumulating into `rows` accumulators. That is
//! the whole point of the layout -- see docs/plan.md section 2.
//!
//! # Scale groups
//!
//! A scale applies to `group_size` consecutive weights **in stream
//! order**, and is written immediately before them. The engine
//! therefore never seeks to fetch a scale; it arrives just in time.
//! The final group of a stream is short when `group_size` does not
//! divide `rows * cols` evenly. Setting `group_size == rows` gives one
//! scale per column, which lets the engine pre-scale the activation
//! once and keep the inner loop free of multipliers.

#![no_std]

mod model;
mod tiling;
mod wire;

pub use model::{DESCRIPTOR_LEN, Encoding, LayoutError, OpDescriptor};
pub use tiling::{group_count, group_len, stream_index, weight_count};
pub use wire::{parse, render};
