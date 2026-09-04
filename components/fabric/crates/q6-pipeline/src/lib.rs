//! Hardware-shaped cycle model for a serial GGML `Q6_K` decoder.

use core::fmt;
mod decode;
mod machine;
use machine::{State, drive};

/// Packed bytes and decoded values in one `Q6_K` block.
pub const BLOCK_BYTES: usize = 210;
/// Values produced by one `Q6_K` block.
pub const BLOCK_VALUES: usize = 256;
const MAX_BLOCKS: usize = 65_536;

/// Abstract datapath widths. Cycles are not seconds until hardware supplies a clock.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Input FIFO capacity in bytes.
    pub fifo_bytes: usize,
    /// Bytes delivered by storage on each productive fetch cycle.
    pub fetch_bytes_per_cycle: usize,
    /// `Q6_K` values decoded per decoder cycle.
    pub decoder_lanes: usize,
    /// Decoded values applied per MAC cycle.
    pub mac_lanes: usize,
}

/// Cycle counters from the abstract pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cycles {
    /// Total elapsed cycles, with fetch overlapped against compute.
    pub total: u64,
    /// Cycles on which storage delivered bytes.
    pub fetch: u64,
    /// Decoder work cycles.
    pub decode: u64,
    /// Selected-expert MAC work cycles.
    pub mac: u64,
    /// Cycles where the consumer had no complete block.
    pub starved: u64,
    /// Cycles where a full FIFO prevented a fetch.
    pub backpressured: u64,
    /// Maximum modeled FIFO occupancy.
    pub peak_fifo_bytes: usize,
}

/// Exact values and modeled timing for one stream.
#[derive(Debug)]
pub struct Outcome {
    /// Independently decoded values; empty when the expert is unselected.
    pub values: Vec<f32>,
    /// Pipeline cycle counters.
    pub cycles: Cycles,
    /// Packed bytes consumed in order.
    pub bytes: usize,
    /// Per-block timing suitable for a bounded visualization trace.
    pub events: Vec<BlockEvent>,
}

/// Observable timing boundaries for one packed block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockEvent {
    /// Zero-based stream position.
    pub block: usize,
    /// Cycle on which the block left the FIFO.
    pub issued: u64,
    /// Modeled decoder completion cycle.
    pub decoded: u64,
    /// Modeled MAC completion; equal to `decoded` when unselected.
    pub accumulated: u64,
    /// FIFO bytes remaining immediately after issue.
    pub fifo_after_issue: usize,
}

/// Invalid input or configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A configuration field was zero or the FIFO cannot hold a block.
    InvalidConfig,
    /// `Q6_K` input ended partway through a block.
    Truncated,
    /// Input exceeded the bounded model.
    TooManyBlocks,
    /// The optional transport checksum detected changed bytes.
    ChecksumMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::InvalidConfig => "invalid pipeline configuration",
                Self::Truncated => "truncated Q6_K block",
                Self::TooManyBlocks => "Q6_K block limit exceeded",
                Self::ChecksumMismatch => "Q6_K transport checksum mismatch",
            }
        )
    }
}
impl std::error::Error for Error {}

/// Small dependency-free checksum used to guard a modeled transport.
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Runs packed blocks through the hardware-shaped model.
///
/// `selected` controls MAC enable; unselected blocks still cross the input pins.
/// `expected_checksum` models framing outside raw GGUF `Q6_K`, which has no checksum.
///
/// # Errors
/// Rejects invalid sizing, truncation, excessive input, or checksum mismatch.
pub fn run(
    bytes: &[u8],
    selected: bool,
    expected_checksum: Option<u64>,
    config: Config,
) -> Result<Outcome, Error> {
    validate(bytes, expected_checksum, config)?;
    let blocks = bytes.len() / BLOCK_BYTES;
    let state = drive(bytes, selected, config, State::new(blocks, selected));
    Ok(Outcome {
        values: state.values,
        cycles: state.cycles,
        bytes: state.fetched,
        events: state.events,
    })
}

fn validate(bytes: &[u8], expected: Option<u64>, config: Config) -> Result<(), Error> {
    if config.fifo_bytes < BLOCK_BYTES
        || [
            config.fetch_bytes_per_cycle,
            config.decoder_lanes,
            config.mac_lanes,
        ]
        .contains(&0)
    {
        return Err(Error::InvalidConfig);
    }
    if bytes.len() % BLOCK_BYTES != 0 {
        return Err(Error::Truncated);
    }
    if bytes.len() / BLOCK_BYTES > MAX_BLOCKS {
        return Err(Error::TooManyBlocks);
    }
    if expected.is_some_and(|value| value != checksum(bytes)) {
        return Err(Error::ChecksumMismatch);
    }
    Ok(())
}
