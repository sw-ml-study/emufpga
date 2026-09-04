//! The forward-only parameter stream trait.

use crate::error::StreamError;

/// A source of parameter bytes that can only be read forward.
///
/// # The surface is the point
///
/// Two methods, neither of which can express a position. An engine
/// generic over this trait has no vocabulary for random access:
///
/// ```compile_fail
/// use spm_stream::WeightStream;
///
/// // There is no seek to call, so this does not compile.
/// fn engine(stream: &mut impl WeightStream) {
///     stream.seek(64);
/// }
/// ```
///
/// Be honest about the strength of that test: `compile_fail` passes
/// if the snippet fails to compile for **any** reason, so a typo in
/// it would look like a pass. Annotating the expected error code
/// (`compile_fail,E0599`) is supposed to narrow that, but under Rust
/// 2024's merged doctests the code is not enforced -- measured on
/// rustc 1.96.0, where substituting a type error still passed. The
/// doctest above therefore proves only "this shape does not compile".
/// The enforced half of the guarantee is `tests/surface.rs`, which
/// fails to build if the trait's required method set ever changes.
///
/// What a consumer *can* do is advance:
///
/// ```
/// use spm_stream::{StreamError, WeightStream};
///
/// fn drain(stream: &mut impl WeightStream) -> Result<usize, StreamError> {
///     let mut buffer = [0u8; 16];
///     let mut total = 0;
///     loop {
///         let taken = stream.next_block(&mut buffer)?;
///         if taken == 0 {
///             return Ok(total);
///         }
///         total += taken;
///     }
/// }
/// ```
pub trait WeightStream {
    /// Fills `dst` with the next bytes, returning how many were
    /// written.
    ///
    /// A short read is normal -- it means the stream's internal buffer
    /// boundary fell here, not that the stream ended. **Zero** means
    /// the stream is exhausted. Callers that need an exact count use
    /// [`WeightStream::read_exact`].
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the backing store fails.
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError>;

    /// Returns to the start of the stream.
    ///
    /// Valid **between operations only**. Rewinding part way through
    /// an operation is a logic error: it is the one thing that would
    /// turn this back into random access. The research's rotating
    /// parameter store rewinds exactly this way, once per full scan.
    ///
    /// # Errors
    /// Returns [`StreamError::Io`] if the backing store fails.
    fn rewind(&mut self) -> Result<(), StreamError>;

    /// Fills `dst` completely, or fails.
    ///
    /// Still forward-only: this is a loop over [`WeightStream::next_block`],
    /// provided because every structured consumer needs it and would
    /// otherwise write the same loop.
    ///
    /// # Errors
    /// Returns [`StreamError::Truncated`] if the stream ends before
    /// `dst` is full, or [`StreamError::Io`] if the store fails.
    fn read_exact(&mut self, dst: &mut [u8]) -> Result<(), StreamError> {
        let needed = dst.len();
        let mut at = 0;
        while at < needed {
            let taken = self.next_block(&mut dst[at..])?;
            if taken == 0 {
                return Err(StreamError::Truncated {
                    needed,
                    available: at,
                });
            }
            at += taken;
        }
        Ok(())
    }
}

impl<T: WeightStream + ?Sized> WeightStream for Box<T> {
    fn next_block(&mut self, dst: &mut [u8]) -> Result<usize, StreamError> {
        (**self).next_block(dst)
    }

    fn rewind(&mut self) -> Result<(), StreamError> {
        (**self).rewind()
    }
}
