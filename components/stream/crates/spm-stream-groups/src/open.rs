//! Reading the header and stream directory off the front of a stream.

use crate::error::GroupError;
use spm_header::{HEADER_LEN, Header, parse as parse_header};
use spm_layout::{DESCRIPTOR_LEN, OpDescriptor, parse as parse_descriptor};
use spm_stream::WeightStream;

/// Bytes an `f32` scale occupies on the wire.
const SCALE_LEN: usize = 4;

/// Consumes the fixed-size header from the front of `stream`.
///
/// # Errors
/// Returns [`GroupError`] if the stream ends early or the header is
/// malformed.
pub(crate) fn read_header(stream: &mut impl WeightStream) -> Result<Header, GroupError> {
    let mut raw = [0u8; HEADER_LEN];
    stream.read_exact(&mut raw)?;
    Ok(parse_header(&raw)?)
}

/// Consumes `count` descriptors immediately following the header.
///
/// # Errors
/// Returns [`GroupError`] if the stream ends early or a descriptor is
/// malformed.
pub(crate) fn read_directory(
    stream: &mut impl WeightStream,
    count: u32,
) -> Result<Vec<OpDescriptor>, GroupError> {
    let mut raw = [0u8; DESCRIPTOR_LEN];
    let mut descriptors = Vec::with_capacity(count as usize);
    for _ in 0..count {
        stream.read_exact(&mut raw)?;
        descriptors.push(parse_descriptor(&raw)?);
    }
    Ok(descriptors)
}

/// Bytes the largest group in `descriptors` will need.
///
/// Sized once at open so the hot path never allocates, mirroring an
/// FPGA FIFO whose depth is fixed at synthesis rather than grown at
/// runtime.
pub(crate) fn widest_group(descriptors: &[OpDescriptor]) -> usize {
    descriptors
        .iter()
        .map(|d| d.encoding.bytes_for(d.group_size as usize))
        .max()
        .unwrap_or(0)
}

/// Reads one scale then `bytes` of payload into `buffer`.
///
/// A free function rather than a method so `GroupStream` keeps room
/// for `rewind`; it needs nothing but the stream and somewhere to put
/// the bytes.
///
/// # Errors
/// Returns [`GroupError`] if the stream ends inside the group.
pub(crate) fn read_payload(
    stream: &mut impl WeightStream,
    buffer: &mut [u8],
) -> Result<f32, GroupError> {
    let mut scale = [0u8; SCALE_LEN];
    stream.read_exact(&mut scale)?;
    stream.read_exact(buffer)?;
    Ok(f32::from_le_bytes(scale))
}
