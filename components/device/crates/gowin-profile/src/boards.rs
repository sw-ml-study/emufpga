//! The five Tang Nano boards, with every figure's provenance.
//!
//! Sources are the board vendor's own documentation. Gowin's
//! DS117-3.2.5E was retrieved and is the authority for the GW1NR
//! family, but its resource tables are CID-encoded and did not
//! extract to text on this machine, so figures come from the Sipeed
//! wiki pages that restate them.

use crate::figure::{Figure, Source};
use crate::model::{BulkMemory, DeviceProfile};

/// Shorthand for a sourced figure.
const fn known(value: u64, source: Source) -> Figure {
    Figure::Known { value, source }
}

const RETRIEVED: &str = "2026-08-30";

const NANO: Source = Source {
    document: "Sipeed Wiki, Tang Nano",
    url: "https://wiki.sipeed.com/hardware/en/tang/Tang-Nano/Nano.html",
    retrieved: RETRIEVED,
};
const NANO4: Source = Source {
    document: "Sipeed Wiki, Tang Nano 4K",
    url: "https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-4K/Nano-4K.html",
    retrieved: RETRIEVED,
};
const NANO9: Source = Source {
    document: "Sipeed Wiki, Tang Nano 9K",
    url: "https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-9K/Nano-9K.html",
    retrieved: RETRIEVED,
};
const NANO20: Source = Source {
    document: "Sipeed Wiki, Tang Nano 20K",
    url: "https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html",
    retrieved: RETRIEVED,
};
const PRIMER25: Source = Source {
    document: "Sipeed Wiki, Tang Primer 25K (GW5A-25 fabric figures)",
    url: "https://wiki.sipeed.com/hardware/en/tang/tang-primer-25k/primer-25k.html",
    retrieved: RETRIEVED,
};

/// No sustained-bandwidth figure exists in board documentation.
const NO_BANDWIDTH: Figure = Figure::Unknown {
    note: "board docs state capacity, not sustained bandwidth; depends on the memory controller",
};
/// No fabric-wide fmax figure exists in board documentation.
const NO_FMAX: Figure = Figure::Unknown {
    note: "datasheets give per-primitive timing; a fabric fmax needs a real place-and-route",
};

/// Tang Nano (marketed as 1K).
pub const NANO_1K: DeviceProfile = DeviceProfile {
    board: "Tang Nano",
    part: "GW1N-1",
    lut4: known(1152, NANO),
    flip_flops: known(864, NANO),
    bsram_bits: known(72 * 1024, NANO),
    bsram_blocks: known(4, NANO),
    ssram_bits: Figure::Unknown {
        note: "not stated on the Sipeed wiki page",
    },
    dsp_18x18: Figure::Unknown {
        note: "not stated on the Sipeed wiki page; GW1N-1 may have none",
    },
    plls: known(1, NANO),
    user_io: known(41, NANO),
    fmax_mhz: NO_FMAX,
    bulk: BulkMemory {
        kind: "PSRAM",
        bits: known(64 * 1024 * 1024, NANO),
        width_bits: Figure::Unknown {
            note: "not stated on the Sipeed wiki page",
        },
        bandwidth_mbps: NO_BANDWIDTH,
    },
    note: "Smallest fabric on hand. 96Kb user flash, 1.2V core.",
};

/// Tang Nano 4K.
pub const NANO_4K: DeviceProfile = DeviceProfile {
    board: "Tang Nano 4K",
    part: "GW1NSR-LV4CQN48PC6/I5",
    lut4: known(4608, NANO4),
    flip_flops: known(3456, NANO4),
    bsram_bits: known(180 * 1024, NANO4),
    bsram_blocks: Figure::Unknown {
        note: "not stated on the Sipeed wiki page",
    },
    ssram_bits: Figure::Unknown {
        note: "not stated on the Sipeed wiki page",
    },
    dsp_18x18: Figure::Unknown {
        note: "not stated on the Sipeed wiki page",
    },
    plls: known(2, NANO4),
    user_io: known(44, NANO4),
    fmax_mhz: NO_FMAX,
    bulk: BulkMemory {
        kind: "PSRAM (in package)",
        bits: Figure::Unknown {
            note: "wiki says the SiP integrates a PSRAM die but does not state its capacity",
        },
        width_bits: Figure::Unknown {
            note: "not stated on the Sipeed wiki page",
        },
        bandwidth_mbps: NO_BANDWIDTH,
    },
    note: "Hard ARM Cortex-M3 core. Could play the stream-controller role on one chip.",
};

/// Tang Nano 9K. The primary target -- docs/plan.md section 6.
pub const NANO_9K: DeviceProfile = DeviceProfile {
    board: "Tang Nano 9K",
    part: "GW1NR-LV9QN88PC6/I5",
    lut4: known(8640, NANO9),
    flip_flops: known(6480, NANO9),
    bsram_bits: known(468 * 1024, NANO9),
    bsram_blocks: known(26, NANO9),
    ssram_bits: known(17_280, NANO9),
    dsp_18x18: known(20, NANO9),
    plls: known(2, NANO9),
    user_io: Figure::Unknown {
        note: "not stated as a count on the Sipeed wiki page; needs UG119E package/pinout guide",
    },
    fmax_mhz: NO_FMAX,
    bulk: BulkMemory {
        kind: "PSRAM",
        bits: known(64 * 1024 * 1024, NANO9),
        width_bits: Figure::Unknown {
            note: "not stated on the Sipeed wiki page",
        },
        bandwidth_mbps: NO_BANDWIDTH,
    },
    note: "Primary target: most mature open-toolchain support. 608Kb user flash, 32Mbit SPI flash.",
};

/// Tang Nano 20K.
pub const NANO_20K: DeviceProfile = DeviceProfile {
    board: "Tang Nano 20K",
    part: "GW2AR-LV18QN88C8/I7",
    lut4: known(20_736, NANO20),
    flip_flops: known(15_552, NANO20),
    bsram_bits: known(828 * 1024, NANO20),
    bsram_blocks: known(46, NANO20),
    ssram_bits: known(41_472, NANO20),
    dsp_18x18: known(48, NANO20),
    plls: known(2, NANO20),
    user_io: Figure::Unknown {
        note: "wiki states 8 I/O banks, not a pin count",
    },
    fmax_mhz: NO_FMAX,
    bulk: BulkMemory {
        kind: "SDR SDRAM",
        bits: known(64 * 1024 * 1024, NANO20),
        width_bits: known(32, NANO20),
        bandwidth_mbps: NO_BANDWIDTH,
    },
    note: "Widest bulk memory interface on hand at 32 bits, and the most LUT4s of the pre-25K parts.",
};

/// Tang Nano 25K.
///
/// Fabric figures are the GW5A-25's, cited from the Tang Primer 25K
/// page, which is the Sipeed documentation that states them. **Which
/// 25K board is on hand is unconfirmed** -- Sipeed sells both a Tang
/// Nano 25K and a Tang Primer 25K -- so every board-level field here
/// is Unknown rather than borrowed from the Primer.
pub const NANO_25K: DeviceProfile = DeviceProfile {
    board: "Tang Nano 25K (board variant unconfirmed)",
    part: "GW5A-25 (Primer 25K carries GW5A-LV25MG121C1/I0)",
    lut4: known(23_040, PRIMER25),
    flip_flops: Figure::Unknown {
        note: "not stated on the Primer 25K page",
    },
    bsram_bits: known(1008 * 1024, PRIMER25),
    bsram_blocks: Figure::Unknown {
        note: "not stated on the Primer 25K page",
    },
    ssram_bits: known(180 * 1024, PRIMER25),
    dsp_18x18: known(28, PRIMER25),
    plls: known(6, PRIMER25),
    user_io: Figure::Unknown {
        note: "Primer 25K page states 8 I/O banks, not a pin count",
    },
    fmax_mhz: NO_FMAX,
    bulk: BulkMemory {
        kind: "unconfirmed",
        bits: Figure::Unknown {
            note: "board variant unconfirmed; Primer 25K takes an external SDRAM module",
        },
        width_bits: Figure::Unknown {
            note: "board variant unconfirmed",
        },
        bandwidth_mbps: NO_BANDWIDTH,
    },
    note: "Arora V family. Largest fabric on hand but least mature open-toolchain support.",
};

/// Every profile, smallest fabric first.
#[must_use]
pub const fn all() -> [DeviceProfile; 5] {
    [NANO_1K, NANO_4K, NANO_9K, NANO_20K, NANO_25K]
}
