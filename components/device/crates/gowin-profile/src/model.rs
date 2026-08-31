//! What a device profile records.

use crate::figure::Figure;

/// On-board bulk memory: the parameter store a board can actually
/// stream from.
///
/// The most important part of a profile for this project. Fabric
/// resources decide whether an engine fits; bulk memory bandwidth
/// decides whether it has anything to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkMemory {
    /// Technology as the board documents it, or `"none"`.
    pub kind: &'static str,
    /// Capacity in bits.
    pub bits: Figure,
    /// Interface width in bits.
    pub width_bits: Figure,
    /// Sustained sequential bandwidth in megabytes per second.
    ///
    /// Unknown for every board in this crate. Board documentation
    /// states capacity and sometimes width, never sustained
    /// bandwidth, which depends on the memory controller as much as
    /// on the part. Measuring it is hardware work.
    pub bandwidth_mbps: Figure,
}

/// One board, its part, and its resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Board name as Sipeed sells it.
    pub board: &'static str,
    /// Full Gowin part number.
    pub part: &'static str,
    /// Four-input lookup tables.
    pub lut4: Figure,
    /// Flip-flops / registers.
    pub flip_flops: Figure,
    /// Block SRAM, total bits.
    pub bsram_bits: Figure,
    /// Block SRAM, number of blocks.
    pub bsram_blocks: Figure,
    /// Shadow / distributed SRAM, bits.
    pub ssram_bits: Figure,
    /// 18x18 multiplier blocks.
    pub dsp_18x18: Figure,
    /// Phase-locked loops.
    pub plls: Figure,
    /// User I/O pins.
    pub user_io: Figure,
    /// Achievable fabric fmax in MHz.
    ///
    /// Unknown for every board. Datasheets give per-primitive timing,
    /// not a fabric-wide figure; the honest source is a real
    /// place-and-route, which is saga 6.
    pub fmax_mhz: Figure,
    /// On-board bulk memory.
    pub bulk: BulkMemory,
    /// Anything else worth knowing about the board.
    pub note: &'static str,
}

impl DeviceProfile {
    /// Names of the fields that could not be sourced.
    ///
    /// Used by tests and by step 008's fit model, which must refuse to
    /// report a utilization it cannot compute rather than substitute
    /// a default.
    #[must_use]
    pub fn unknown_fields(&self) -> Vec<&'static str> {
        [
            ("lut4", self.lut4),
            ("flip_flops", self.flip_flops),
            ("bsram_bits", self.bsram_bits),
            ("bsram_blocks", self.bsram_blocks),
            ("ssram_bits", self.ssram_bits),
            ("dsp_18x18", self.dsp_18x18),
            ("plls", self.plls),
            ("user_io", self.user_io),
            ("fmax_mhz", self.fmax_mhz),
            ("bulk.bits", self.bulk.bits),
            ("bulk.width_bits", self.bulk.width_bits),
            ("bulk.bandwidth_mbps", self.bulk.bandwidth_mbps),
        ]
        .into_iter()
        .filter(|(_, figure)| !figure.is_known())
        .map(|(name, _)| name)
        .collect()
    }

    /// The fabric figures a fit model needs to place an engine:
    /// LUT4, flip-flops, block SRAM and DSP.
    ///
    /// Separate from [`DeviceProfile::unknown_fields`] because these
    /// four decide whether a design *fits*, while fmax and bulk
    /// bandwidth decide how fast it *runs*. A board can be usable for
    /// the first question and unanswerable for the second, which is
    /// exactly the situation every board here is in.
    #[must_use]
    pub fn fabric_is_complete(&self) -> bool {
        self.lut4.is_known()
            && self.flip_flops.is_known()
            && self.bsram_bits.is_known()
            && self.dsp_18x18.is_known()
    }
}
