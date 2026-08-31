//! Pulling scale groups off a parameter stream, one at a time.

use crate::error::GroupError;
use crate::open::{read_directory, read_header, read_payload, widest_group};
use spm_header::{HEADER_LEN, Header};
use spm_layout::{DESCRIPTOR_LEN, Encoding, OpDescriptor};
use spm_stream::WeightStream;
use spm_walk::Cursor;

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
    /// How `packed` is encoded.
    ///
    /// Carried on the view so a consumer cannot decode a group with
    /// the wrong codec. Before this existed the discriminant was
    /// written into every descriptor and read by nobody: a bf16
    /// stream would have been decoded as f32 and produced plausible
    /// garbage rather than an error.
    pub encoding: Encoding,
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
    /// Header plus directory, in bytes. Skipped on every rewind.
    prologue: usize,
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
        let prologue = HEADER_LEN + descriptors.len() * DESCRIPTOR_LEN;
        Ok(Self {
            stream,
            header,
            descriptors,
            cursor,
            buffer,
            prologue,
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
        let encoding = self.descriptors[stream_index].encoding;
        let bytes = encoding.bytes_for(count as usize);
        self.cursor.advance(&self.descriptors);
        let payload = read_payload(&mut self.stream, &mut self.buffer[..bytes]);
        Some(payload.map(move |scale| GroupView {
            stream: stream_index,
            scale,
            count,
            encoding,
            packed: &self.buffer[..bytes],
        }))
    }

    /// Returns to the **first group**, not to byte zero.
    ///
    /// The header and directory sit ahead of the payload, so a bare
    /// stream rewind would land on metadata. This skips them and
    /// resets the cursor, leaving the reader exactly where `open` left
    /// it.
    ///
    /// Legal **between operations only**, which is the same contract
    /// [`spm_stream::WeightStream::rewind`] carries. A recursive model
    /// rewinds once per pass over its rotating region -- that is what
    /// makes the region rotate, and it is why a consumption-order
    /// layout puts that region first.
    ///
    /// # Errors
    /// Returns [`GroupError`] if the stream cannot be rewound or ends
    /// inside its own prologue.
    pub fn rewind(&mut self) -> Result<(), GroupError> {
        self.stream.rewind()?;
        let mut left = self.prologue;
        while left > 0 {
            let take = left.min(self.buffer.len().max(1));
            self.stream.read_exact(&mut self.buffer[..take])?;
            left -= take;
        }
        self.cursor = Cursor::new(&self.descriptors);
        Ok(())
    }
}
