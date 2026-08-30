//! The scan: consume a parameter stream, accumulate, report metrics.

use crate::datapath::apply_group;
use crate::error::GemvError;
use spm_accum::AccumulatorBank;
use spm_activations::Activations;
use spm_stream::WeightStream;
use spm_stream_groups::{GroupStream, GroupView};
use spm_stream_metrics::ScanMetrics;
use std::time::Instant;

/// What one full scan produced.
#[derive(Debug)]
pub struct GemvOutcome {
    /// Accumulators, one bank per batch lane.
    pub bank: AccumulatorBank,
    /// Counters and derived rates for the scan.
    pub metrics: ScanMetrics,
}

/// Mutable state carried across one scan.
struct Scan<'a> {
    bank: &'a mut AccumulatorBank,
    activations: &'a Activations,
    metrics: &'a mut ScanMetrics,
    scaled: Vec<f32>,
    rows: usize,
    position: usize,
}

/// Runs the first operation in `stream` against `activations`.
///
/// Saga 1 handles one operation per file; multi-stream scheduling is
/// saga 7's `MoE` work.
///
/// # Errors
/// Returns [`GemvError`] if the stream is malformed, declares no
/// operations, or the caller supplied too few activations.
pub fn run_gemv(
    stream: impl WeightStream,
    activations: &Activations,
) -> Result<GemvOutcome, GemvError> {
    let mut groups = GroupStream::open(stream)?;
    let (rows, mut metrics) = prepare(&groups, activations)?;
    let mut bank = AccumulatorBank::new(activations.lanes, rows);
    let mut scan = Scan {
        bank: &mut bank,
        activations,
        metrics: &mut metrics,
        scaled: vec![0.0f32; activations.lanes],
        rows,
        position: 0,
    };
    drive(&mut groups, &mut scan)?;
    Ok(GemvOutcome { bank, metrics })
}

/// Validates the operation against the activations and seeds the
/// counters, before a single byte of payload is consumed.
///
/// Checked up front on purpose: the activations are resident and known
/// in advance, so a shape mismatch should never be discovered part way
/// through a stream that cannot be rewound.
fn prepare(
    groups: &GroupStream<impl WeightStream>,
    activations: &Activations,
) -> Result<(usize, ScanMetrics), GemvError> {
    let descriptor = *groups.descriptors.first().ok_or(GemvError::NoStreams)?;
    let (rows, cols) = (descriptor.rows as usize, descriptor.cols as usize);
    if activations.cols < cols {
        return Err(GemvError::MissingActivations {
            needed: cols,
            supplied: activations.cols,
        });
    }
    let metrics = ScanMetrics {
        resident_parameter_bytes: groups.resident_parameter_bytes() as u64,
        total_parameter_bytes: (rows as u64 * cols as u64).div_ceil(4),
        ..ScanMetrics::default()
    };
    Ok((rows, metrics))
}

/// Pulls groups until the first stream ends.
fn drive(
    groups: &mut GroupStream<impl WeightStream>,
    scan: &mut Scan<'_>,
) -> Result<(), GemvError> {
    loop {
        let started = Instant::now();
        let next = groups.next_group().transpose()?;
        scan.metrics.storage_time += started.elapsed();
        // Timed before the checks below, so a scan that stops at a
        // stream boundary still reports the read it paid for.
        let Some(group) = next else { return Ok(()) };
        if group.stream != 0 {
            return Ok(());
        }
        scan.consume(&group);
    }
}

impl Scan<'_> {
    /// Applies one group and folds its cost into the counters.
    ///
    /// Applications count every weight presented to every lane,
    /// zeros included: a zero still occupies a slot in the stream and
    /// a cycle in the engine. That keeps `Ps` a measure of reuse
    /// rather than of sparsity, which the format does not yet exploit.
    fn consume(&mut self, group: &GroupView<'_>) {
        let count = group.count as usize;
        let started = Instant::now();
        apply_group(
            self.bank,
            self.activations,
            &mut self.scaled,
            (group.scale, group.packed, count),
            (self.position, self.rows),
        );
        self.metrics.compute_time += started.elapsed();
        self.metrics.parameter_bytes_read += group.packed.len() as u64;
        self.metrics.weights_decoded += count as u64;
        self.metrics.weight_applications += (count * self.bank.lanes) as u64;
        self.position += count;
    }
}
