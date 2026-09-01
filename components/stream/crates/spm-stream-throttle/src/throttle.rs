//! The rate limiter, and the stall it accumulates.

use spm_stream_metrics::widen;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Wraps a stream and delivers its bytes at `bytes_per_second`.
///
/// The limit is enforced against a **virtual clock**: after `n` bytes,
/// no read may return before `n / rate` has elapsed since the first
/// read. Pacing each read individually would measure sleep
/// granularity rather than bandwidth -- at 200 MB/s a 4 KiB group is
/// 20 microseconds, far below what a sleep can resolve -- so the
/// deadline is cumulative and the wait is spun when it is short.
pub struct Throttle<S> {
    pub(crate) inner: S,
    rate: f64,
    started: Option<Instant>,
    pub(crate) served: u64,
    stalled: Arc<AtomicU64>,
}

impl<S> Throttle<S> {
    /// Limits `inner` to `bytes_per_second`.
    ///
    /// A rate of zero or less means unlimited, so a caller can sweep a
    /// range including "no limit" without a special case.
    #[must_use]
    pub fn new(inner: S, bytes_per_second: f64) -> Self {
        Self {
            inner,
            rate: bytes_per_second,
            started: None,
            served: 0,
            stalled: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A handle to the stall counter, readable after this stream has
    /// been moved into a consumer.
    ///
    /// `GroupStream` takes ownership of the stream it reads, so a
    /// caller that wants the stall time must take this first. The
    /// alternative -- an accessor on `GroupStream` -- would put a
    /// measurement concern into the type whose whole job is to expose
    /// nothing but forward reads.
    #[must_use]
    pub fn meter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.stalled)
    }

    /// How long the engine spent waiting on the store.
    ///
    /// The measurement this crate exists for. Against total wall
    /// clock it is `eta` observed rather than modelled: near zero
    /// means compute-bound, approaching the whole run means
    /// store-bound.
    #[must_use]
    pub fn stalled(&self) -> Duration {
        Duration::from_nanos(self.stalled.load(Ordering::Relaxed))
    }

    /// Waits until the virtual clock has caught up with `served`.
    pub(crate) fn pace(&mut self) {
        if self.rate <= 0.0 {
            return;
        }
        let started = *self.started.get_or_insert_with(Instant::now);
        let deadline = started + Duration::from_secs_f64(widen(self.served) / self.rate);
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let wait = deadline - now;
        // Sleep for long waits, spin for short ones: at 200 MB/s a
        // 4 KiB group is 20us, well under what a sleep can resolve.
        if wait > Duration::from_micros(500) {
            std::thread::sleep(wait);
        } else {
            while Instant::now() < deadline {}
        }
        self.stalled.fetch_add(
            u64::try_from(wait.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }
}
