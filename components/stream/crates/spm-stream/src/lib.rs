//! The sequential parameter stream.
//!
//! [`WeightStream`] is stated as a **restriction, not a capability**.
//! Its whole surface is:
//!
//! ```text
//! next_block(&mut self, dst: &mut [u8]) -> Result<usize>
//! rewind(&mut self)                     -- between operations only
//! ```
//!
//! There is no seek, no addressing, no random access, and no way to
//! ask where the cursor is. Per docs/plan.md section 3, that is the
//! experimental rule the whole project rests on: the tensor engine may
//! never seek backward or randomly into the parameter stream while
//! performing an operation.
//!
//! # What the type system actually guarantees
//!
//! Be precise about this, because it is easy to overclaim. A concrete
//! backing store obviously *can* seek -- a file has `Seek`, a slice
//! has indexing. What the trait guarantees is that **code written
//! against `impl WeightStream` cannot reach those methods**. An engine
//! generic over this trait has no vocabulary for random access, so the
//! architecture cannot quietly erode back into a memory controller as
//! the code grows. That is a guarantee about consumers, not about
//! implementations, and it is the one that matters: implementations
//! are few and reviewed, consumers are many and will be written by
//! future sessions.
//!
//! The `compile_fail` doctest on [`WeightStream`] pins it.
//!
//! # `rewind` is not a seek
//!
//! [`WeightStream::rewind`] returns to the start of the whole stream,
//! and only between operations. It cannot move to an arbitrary
//! position, and calling it mid-operation is a logic error the engine
//! is responsible for not committing -- the research's rotating
//! parameter store rewinds exactly this way, once per full scan.

mod error;
mod stream;

pub use error::StreamError;
pub use stream::WeightStream;
