//! Ternary weight packing for the `.spm` physical execution layout.
//!
//! Two bits per weight, four codes, chosen so the FPGA decoder is two
//! wires rather than a lookup table:
//!
//! | code | bit 1 | bit 0 | meaning |
//! |------|-------|-------|---------|
//! | `00` |     0 |     0 | `0`     |
//! | `01` |     0 |     1 | `+1`    |
//! | `11` |     1 |     1 | `-1`    |
//! | `10` |     1 |     0 | invalid |
//!
//! Bit 0 is "nonzero": it gates the accumulator enable. Bit 1 is
//! "negative": it selects subtract over add. So a weight arriving off
//! the stream drives the arithmetic unit directly, with no decode
//! stage in between -- the storage stream *is* the instruction stream.
//!
//! Code `10` means "negative zero", which is not a value this encoding
//! can produce. It is left permanently invalid rather than assigned a
//! meaning, so the hardware decoder stays combinational and stateless.
//! Readers MUST reject it; it is the cheapest corruption check the
//! format has.
//!
//! Weights pack four to a byte, least significant pair first: weight
//! `k` of a group occupies bits `2k` and `2k+1` of byte `k / 4`.
//! LSB-first because a bit-serial consumer (an FPGA shift register, or
//! an RP2350 PIO state machine) receives the low bit first.
//!
//! This crate is `no_std` and allocation-free: encoding writes into a
//! caller-provided slice. Front 3 (RP2350) needs both, and buffer
//! discipline here mirrors the FIFO discipline in hardware.

#![no_std]

mod error;
mod model;
mod pack;

pub use error::CodecError;
pub use model::{NEGATIVE_BIT, NONZERO_BIT, Ternary};
pub use pack::{code_at, decode_into, encode_into, packed_len};
