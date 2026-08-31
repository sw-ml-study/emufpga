//! Consumption order: which stream comes first, and which get rewound.
//!
//! # The mistake this crate exists to prevent
//!
//! `scripts/extract-checkpoint` lists tensors alphabetically, because
//! that is what `sorted()` does. For TRM that gives, per layer,
//! `down_proj, gate_up_proj, o_proj, qkv_proj` -- the exact reverse of
//! the execution order `qkv_proj, o_proj, gate_up_proj, down_proj`. A
//! forward sweep of such a file would have to seek backward, which is
//! the one thing this architecture forbids.
//!
//! docs/research.txt puts it plainly: arrange weights physically in
//! exactly the order the tensor engine consumes them. Alphabetical is
//! not that order for any model, and the first import got it wrong.
//!
//! # Rotating first
//!
//! An order file has two sections. `[rotating]` is swept once per
//! operation and rewound; `[resident]` is read once into RAM.
//!
//! Rotating **must** come first, because `rewind` returns to the start
//! of the stream rather than to an arbitrary point -- there is no seek
//! to offer it one. Anything ahead of the rotating region would be
//! re-read on every sweep for nothing.
//!
//! For TRM the split is stark: 8 matrices totalling 6,815,744 weights
//! rotate, and 7 tensors totalling 8,706 weights stay resident. The
//! resident part is 0.13% of the model, which is why streaming it
//! would be silly -- docs/plan.md section 3 puts small state in
//! ordinary memory on purpose.

mod order;

pub use order::{Order, apply_order, parse_order, reorder};
