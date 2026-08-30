//! Reading a `.spm` file forward, one scale group at a time.

use crate::error::FileError;
use spm_codec::packed_len;
use spm_header::{HEADER_LEN, Header, parse as parse_header};
use spm_layout::{DESCRIPTOR_LEN, OpDescriptor, parse as parse_descriptor};
use spm_walk::Cursor;

/// Bytes an `f32` scale occupies on the wire.
const SCALE_LEN: usize = 4;

/// One scale group as it arrives off the stream.
///
/// `packed` is borrowed and still packed: the consumer receives bytes
/// and unpacks them itself, exactly as the FPGA does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Group<'a> {
    /// Index of the stream this group belongs to.
    pub stream: usize,
    /// Scale applied to every weight in the group.
    pub scale: f32,
    /// Weights in the group. Short for a stream's final group.
    pub count: u32,
    /// The packed weight bytes, byte-aligned.
    pub packed: &'a [u8],
}

/// A forward-only reader over a `.spm` file held in memory.
///
/// `header` and `descriptors` are parsed eagerly and are ordinary
/// random-access memory -- they are metadata, which docs/plan.md
/// section 3 explicitly allows. The payload is not: it is reachable
/// only through [`SpmReader::next_group`], which moves forward.
pub struct SpmReader<'a> {
    /// The file header.
    pub header: Header,
    /// The stream directory.
    pub descriptors: Vec<OpDescriptor>,
    payload: &'a [u8],
    at: usize,
    cursor: Cursor,
}

impl<'a> SpmReader<'a> {
    /// Parses the header and stream directory of `src`.
    ///
    /// # Errors
    /// Returns [`FileError`] if the header or any descriptor is
    /// malformed, or the file ends inside the directory.
    pub fn parse(src: &'a [u8]) -> Result<Self, FileError> {
        let header = parse_header(src)?;
        let mut descriptors = Vec::new();
        let mut at = HEADER_LEN;
        for _ in 0..header.stream_count {
            descriptors.push(parse_descriptor(src.get(at..).unwrap_or_default())?);
            at += DESCRIPTOR_LEN;
        }
        let cursor = Cursor::new(&descriptors);
        Ok(Self {
            header,
            descriptors,
            payload: src.get(at..).unwrap_or_default(),
            at: 0,
            cursor,
        })
    }

    /// Advances to the next scale group, or `None` at end of payload.
    ///
    /// # Errors
    /// Returns [`FileError::PayloadTruncated`] if the file ends inside
    /// a group.
    pub fn next_group(&mut self) -> Option<Result<Group<'a>, FileError>> {
        let count = self.cursor.group_len(&self.descriptors)?;
        let stream = self.cursor.stream;
        Some(
            self.take(count)
                .inspect(|_| {
                    self.cursor.advance(&self.descriptors);
                })
                .map(|(scale, packed)| Group {
                    stream,
                    scale,
                    count,
                    packed,
                }),
        )
    }

    /// Consumes one scale plus `count` packed weights at the cursor.
    fn take(&mut self, count: u32) -> Result<(f32, &'a [u8]), FileError> {
        let needed = SCALE_LEN + packed_len(count as usize);
        let available = self.payload.len().saturating_sub(self.at);
        let raw = self
            .payload
            .get(self.at..self.at + needed)
            .ok_or(FileError::PayloadTruncated { needed, available })?;
        self.at += needed;
        let scale = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok((scale, &raw[SCALE_LEN..]))
    }
}
