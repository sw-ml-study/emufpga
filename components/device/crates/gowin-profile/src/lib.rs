//! Gowin device profiles for the five Tang Nano boards on hand.
//!
//! # Provenance is part of the data
//!
//! docs/plan.md section 6 originally carried a table of figures
//! written from memory and marked UNVERIFIED. This crate replaces it.
//! Every number here carries the document it came from, in the code
//! rather than only in a doc, and anything that could not be sourced
//! is [`Figure::Unknown`] with a note saying what was tried.
//!
//! [`Figure`] exists so an unknown cannot be silently read as a
//! number. There is no `unwrap_or(0)` path: a consumer must decide
//! what to do about a missing value, which for step 008's fit model
//! means refusing to report a utilization it cannot compute.
//!
//! # What could not be sourced, and why it matters
//!
//! **Bulk memory bandwidth is Unknown for every board.** That is the
//! most consequential gap in this crate. The architecture's whole
//! premise is trading random access for cheap sequential bandwidth,
//! and step 006 measured the CPU reference as roughly 196x too slow
//! to saturate even a page-cached read (docs/results.md). Which side
//! of that ratio a given board lands on is decided by the bandwidth
//! its PSRAM or SDRAM can actually sustain -- a number the board
//! documentation does not state, and one that depends on the memory
//! controller as much as the part.
//!
//! **Achievable fmax is Unknown for every board.** Datasheets give
//! per-primitive timing, not a fabric-wide figure, and the honest
//! source for this is a real place-and-route (saga 6).
//!
//! Both gaps are load-bearing for step 008. A fit model that quietly
//! assumed values for them would produce numbers nobody could check.

mod boards;
mod figure;
mod model;

pub use boards::{NANO_1K, NANO_4K, NANO_9K, NANO_20K, NANO_25K, all};
pub use figure::{Figure, Source};
pub use model::{BulkMemory, DeviceProfile};
