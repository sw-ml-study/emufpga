//! Executing a `.spm` file through the modelled pipeline.

use crate::config::{FabricConfig, FabricError, malformed};
use crate::cycles::{FabricOutcome, Pipeline};
use spm_accum::AccumulatorBank;
use spm_activations::Activations;
use spm_codec::{NEGATIVE_BIT, NONZERO_BIT, code_at};
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// Runs `stream` through the modelled pipeline.
///
/// # Errors
/// Returns [`FabricError`] if the configuration cannot describe a
/// working pipeline, or the stream is malformed or declares no
/// operations.
pub fn run_fabric(
    stream: impl WeightStream,
    activations: &Activations,
    config: &FabricConfig,
) -> Result<FabricOutcome, FabricError> {
    config.validate()?;
    let mut groups = GroupStream::open(stream).map_err(malformed)?;
    let first = *groups.descriptors.first().ok_or(FabricError::NoStreams)?;
    let (bank, pipeline, weights) = scan(&mut groups, activations, (config, first))?;
    Ok(FabricOutcome {
        bank,
        pipeline,
        weights,
    })
}

/// Accumulators, resident activations, and the scaled-activation
/// register the datapath writes each time it crosses a column.
///
/// Bundled so the inner loop reads as one call rather than a
/// five-argument invocation wrapped over seven lines.
struct Datapath<'a> {
    bank: AccumulatorBank,
    activations: &'a Activations,
    scaled: Vec<f32>,
}

impl<'a> Datapath<'a> {
    /// A zeroed datapath for `rows` outputs.
    fn new(activations: &'a Activations, rows: usize) -> Self {
        Self {
            bank: AccumulatorBank::new(activations.lanes, rows),
            activations,
            scaled: vec![0.0f32; activations.lanes],
        }
    }

    /// Applies one scale group. Identical arithmetic to
    /// `spm-gemv-ref`, written out separately on purpose: the
    /// differential test then compares two implementations of the
    /// rule rather than one implementation against itself.
    ///
    /// `NONZERO_BIT` is the accumulator enable and `NEGATIVE_BIT` the
    /// add/subtract select. No multiplier appears between them.
    fn apply(&mut self, group: (f32, &[u8], usize), at: (usize, usize)) {
        let (scale, packed, count) = group;
        let (position, rows) = at;
        let mut current_col = usize::MAX;
        for local in 0..count {
            let Some(code) = code_at(packed, local) else {
                break;
            };
            let index = position + local;
            let col = index / rows;
            if col != current_col {
                self.activations.scale_column(scale, col, &mut self.scaled);
                current_col = col;
            }
            if code & NONZERO_BIT != 0 {
                self.bank
                    .accumulate(index % rows, code & NEGATIVE_BIT != 0, &self.scaled);
            }
        }
    }
}

/// Walks every group of the first stream through the pipeline.
fn scan(
    groups: &mut GroupStream<impl WeightStream>,
    activations: &Activations,
    shape: (&FabricConfig, spm_layout::OpDescriptor),
) -> Result<(AccumulatorBank, Pipeline, u64), FabricError> {
    let (config, descriptor) = shape;
    let rows = descriptor.rows as usize;
    let mut dp = Datapath::new(activations, rows);
    let mut pipeline = Pipeline::start(config);
    let (mut at, mut weights) = (0usize, 0u64);
    while let Some(group) = groups.next_group() {
        let g = group.map_err(malformed)?;
        if g.stream != 0 {
            break;
        }
        let count = g.count as usize;
        dp.apply((g.scale, g.packed, count), (at, rows));
        let group = (count, descriptor.encoding.bytes_for(count));
        pipeline.process(group, activations.lanes, config);
        (at, weights) = (at + count, weights + count as u64);
    }
    Ok((dp.bank, pipeline, weights))
}
