//! Pulling scale groups off a parameter stream, one at a time.

use crate::error::GroupError;
use crate::open::{read_directory, read_header, widest_group};
use spm_codec::packed_len;
use spm_header::Header;
use spm_layout::OpDescriptor;
use spm_stream::WeightStream;
use spm_walk::Cursor;

/// Bytes an `f32` scale occupies on the wire.
const SCALE_LEN: usize = 4;

/// One scale group as it arrives, borrowed from the reusable buffer.
///
/// `packed` is still packed. The consumer receives bytes and unpacks
/// them itself, exactly as the FPGA does.
#[derive(Debug, PartialEq)]
pub struct GroupView<'a> {
    /// Index of the stream this group belongs to.
    pub stream: usize,
    /// Scale applied to every weight in the group.
    pub scale: f32,
    /// Weights in the group. Short for a stream's final group.
    pub count: u32,
    /// The packed weight bytes.
    pub packed: &'a [u8],
}

/// Reads a `.spm` structure off a forward-only parameter stream.
pub struct GroupStream<S> {
    stream: S,
    /// The file header. Metadata, so RAM-resident by design.
    pub header: Header,
    /// The stream directory. Metadata, so RAM-resident by design.
    pub descriptors: Vec<OpDescriptor>,
    cursor: Cursor,
    buffer: Vec<u8>,
}

impl<S: WeightStream> GroupStream<S> {
    /// Consumes the header and directory from the front of `stream`,
    /// leaving it positioned at the first group.
    ///
    /// # Errors
    /// Returns [`GroupError`] if the stream ends early or the header
    /// or a descriptor is malformed.
    pub fn open(mut stream: S) -> Result<Self, GroupError> {
        let header = read_header(&mut stream)?;
        let descriptors = read_directory(&mut stream, header.stream_count)?;
        let cursor = Cursor::new(&descriptors);
        let buffer = vec![0; widest_group(&descriptors)];
        Ok(Self {
            stream,
            header,
            descriptors,
            cursor,
            buffer,
        })
    }

    /// Parameter bytes held in random-access memory at any moment.
    ///
    /// One group's packed weights, never the model. This is the
    /// numerator of `Rp` in docs/plan.md section 4, reported rather
    /// than assumed so the residency claim can be measured.
    #[must_use]
    pub fn resident_parameter_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Advances to the next group, or `None` past the last stream.
    ///
    /// # Errors
    /// Returns [`GroupError::Stream`] if the payload ends inside a
    /// group.
    pub fn next_group(&mut self) -> Option<Result<GroupView<'_>, GroupError>> {
        let count = self.cursor.group_len(&self.descriptors)?;
        let stream_index = self.cursor.stream;
        self.cursor.advance(&self.descriptors);
        Some(self.take(count).map(move |scale| GroupView {
            stream: stream_index,
            scale,
            count,
            packed: &self.buffer[..packed_len(count as usize)],
        }))
    }

    /// Reads one scale then `count` packed weights into the buffer.
    fn take(&mut self, count: u32) -> Result<f32, GroupError> {
        let mut scale = [0u8; SCALE_LEN];
        self.stream.read_exact(&mut scale)?;
        let bytes = packed_len(count as usize);
        self.stream.read_exact(&mut self.buffer[..bytes])?;
        Ok(f32::from_le_bytes(scale))
    }
}
