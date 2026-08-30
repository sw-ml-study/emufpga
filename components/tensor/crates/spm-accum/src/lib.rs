//! Accumulator banks with a batch dimension.
//!
//! One bank of `rows` accumulators per batch lane. The batch dimension
//! is the entire point: a weight arriving off the stream is applied to
//! **every** lane before it is discarded, which is what raises `Ps`
//! from 1 to the batch size. Storage bandwidth is fixed; reuse is the
//! only free variable.
//!
//! # Why `f32` and not `i32`
//!
//! The `Ternary2F32I32` profile name says `i32` accumulators, and the
//! research's sketch is integer: `-1 => acc -= x`, `+1 => acc += x`.
//! Implementing it revealed that the profile name settled a question
//! that had not actually been worked out, so it is worth writing down.
//!
//! Scale groups run along the **stream**, and the stream is
//! column-major, so a group's scale can vary with both the output row
//! and the input column. That leaves two ways to keep the inner loop
//! free of multipliers:
//!
//! 1. **Pre-scale the activation.** Compute `s * x[j]` once per
//!    (group, column) pair and add or subtract it. One multiply per
//!    column when `group_size == rows`; none in the inner loop. The
//!    accumulator has to be `f32`.
//! 2. **Pre-scale into a fixed-point integer.** Round `s * x[j] * K`
//!    to an integer once per (group, column) and accumulate in `i32`.
//!    Also multiplier-free in the inner loop, and cheaper in LUTs, but
//!    it introduces rounding error.
//!
//! Both are real designs and the choice belongs to saga 2, when the
//! fabric exists to measure them. This crate is the **correctness
//! oracle** for that comparison, so it takes option 1: `f32`
//! accumulation is exact, and an oracle that carries its own
//! quantization error cannot adjudicate anyone else's. No format
//! change was needed -- the profile discriminant is a wire value and
//! nothing on disk depends on the accumulator width.

mod bank;

pub use bank::AccumulatorBank;
