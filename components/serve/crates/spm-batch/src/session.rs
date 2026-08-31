//! One client's state between decode steps, and the shared scratch.

use spm_kv::KvCache;
use spm_smol::SmolConfig;
use spm_stream::WeightStream;
use spm_stream_groups::GroupStream;

/// A client mid-generation.
///
/// Everything here is per-client and none of it is a weight. This is
/// the half of a serving engine that scales WITH the client count,
/// against the weight traffic that does not.
pub struct Client {
    /// Attention cache for every layer.
    pub cache: KvCache,
    /// The token this client feeds in on the next step.
    pub token: u32,
    /// Tokens produced so far, in order.
    pub produced: Vec<u32>,
}

impl Client {
    /// A client with an empty cache, ready to consume `token`.
    #[must_use]
    pub fn new(config: &SmolConfig, context: usize, token: u32) -> Self {
        Self {
            cache: KvCache::new(config.layers, context, config.kv_width()),
            token,
            produced: Vec::new(),
        }
    }

    /// Bytes this client holds. The bill for serving it.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cache.bytes()
    }
}

/// Buffers shared by every client in one step.
///
/// Sized `clients x width` so the streamed matmuls can treat the
/// client count as the batch dimension: one weight, applied to every
/// client, then discarded. That is the amortization, expressed as a
/// buffer shape.
pub struct Scratch {
    /// Hidden state per client, `clients x hidden`.
    pub states: Vec<f32>,
    pub(crate) normed: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) projected: Vec<f32>,
    /// Next-token logits per client, `clients x vocab`.
    pub logits: Vec<f32>,
}

impl Scratch {
    /// Buffers for `clients` concurrent clients.
    #[must_use]
    pub fn new(config: &SmolConfig, clients: usize) -> Self {
        let (d, i, kv) = (config.hidden, config.intermediate, config.kv_width());
        Self {
            states: vec![0.0; clients * d],
            normed: vec![0.0; clients * d],
            q: vec![0.0; clients * d],
            k: vec![0.0; clients * kv],
            v: vec![0.0; clients * kv],
            attn: vec![0.0; clients * d],
            gate: vec![0.0; clients * i],
            up: vec![0.0; clients * i],
            projected: vec![0.0; clients * d],
            logits: vec![0.0; clients * config.vocab],
        }
    }
}

/// What one decode step consumed and produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepReport {
    /// Clients served by this one sweep.
    pub clients: usize,
    /// Weight streams swept. Constant in `clients` -- the claim.
    pub streams: usize,
    /// Weight bytes read. Also constant in `clients`.
    ///
    /// Taken from the descriptors, which
    /// `declared_widths_account_for_every_payload_byte` asserts are
    /// the authority on width. Computing it from a weight count and an
    /// assumed element size is postmortem 2 defect 11 exactly.
    pub weight_bytes: usize,
}

impl StepReport {
    /// The report for a sweep of `config`'s streams serving `clients`.
    ///
    /// Weight bytes come from the descriptors, which
    /// `declared_widths_account_for_every_payload_byte` asserts are
    /// the authority on width. Deriving them from a weight count and
    /// an assumed element size is postmortem 2 defect 11 exactly.
    #[must_use]
    pub fn for_sweep<S: WeightStream>(
        groups: &GroupStream<S>,
        config: &SmolConfig,
        clients: usize,
    ) -> Self {
        Self {
            clients,
            streams: config.streams(),
            weight_bytes: groups
                .descriptors
                .iter()
                .take(config.streams())
                .map(|d| d.encoding.bytes_for(d.rows as usize * d.cols as usize))
                .sum(),
        }
    }
}
