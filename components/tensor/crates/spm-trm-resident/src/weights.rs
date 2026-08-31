//! Every parameter, in RAM, indexable by stream.

use spm_codec_dense::decode_into;
use spm_linear::{LinearError, resident};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// The whole model held in random-access memory.
///
/// One `Vec<f32>` per stream, in the file's declared order, each in
/// the same column-major layout the stream delivers. Holding the
/// layout fixed is what lets the resident and streamed paths agree bit
/// for bit: any disagreement is then about mechanism, never about how
/// a matrix was read.
///
/// This is the thing the Serial Parameter Machine exists to avoid
/// allocating. Its size is the number docs/results.md reports as the
/// memory axis of the comparison.
pub struct ResidentWeights {
    matrices: Vec<Vec<f32>>,
}

impl ResidentWeights {
    /// Drains `groups` to the end, keeping every weight.
    ///
    /// Reads sequentially because that is the only way a `.spm` can be
    /// read; the random access this type offers begins once loading is
    /// done. A conventional engine pays exactly this cost at startup,
    /// then keeps the result for the life of the process.
    ///
    /// # Errors
    /// Returns [`LinearError::Stream`] if a group fails to decode.
    pub fn load<S: WeightStream>(groups: &mut GroupStream<S>) -> Result<Self, LinearError> {
        let mut matrices = vec![Vec::new(); groups.descriptors.len()];
        while let Some(group) = groups.next_group() {
            let view = group.map_err(|e| LinearError::Stream {
                detail: e.to_string(),
            })?;
            let at = matrices[view.stream].len();
            matrices[view.stream].resize(at + view.count as usize, 0.0);
            decode_into(view.packed, &mut matrices[view.stream][at..]).map_err(|needed| {
                LinearError::Stream {
                    detail: format!("group needed {needed} bytes"),
                }
            })?;
        }
        Ok(Self { matrices })
    }

    /// Applies matrix `index` to a batch, straight out of RAM.
    ///
    /// The subscript is the operation the streamed path cannot
    /// perform. Reaching stream `n` there means having already read
    /// streams `0..n`; here it is an array index, and that capability
    /// is precisely what the resident bytes are buying.
    ///
    /// # Errors
    /// Returns [`LinearError`] if the matrix disagrees with `shape`.
    ///
    /// # Panics
    /// Panics if `index` is past the last stream, which is an engine
    /// bug rather than an input error.
    pub fn project(
        &self,
        index: usize,
        shape: (usize, usize),
        batch: (&[f32], usize),
        out: &mut [f32],
    ) -> Result<(), LinearError> {
        resident(&self.matrices[index], shape, batch, out)
    }

    /// Parameter bytes held in random-access memory, all of them.
    ///
    /// The counterpart to `GroupStream::resident_parameter_bytes`,
    /// which reports one group. Comparing the two is the point of the
    /// whole exercise, so both are measured rather than assumed.
    #[must_use]
    pub fn parameter_bytes(&self) -> usize {
        self.matrices.iter().map(Vec::len).sum::<usize>() * size_of::<f32>()
    }
}
