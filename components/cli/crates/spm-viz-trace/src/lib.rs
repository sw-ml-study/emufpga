//! Bounded, dependency-free trace schema for the serial-MoE visual emulator.

/// One expert visit in physical stream order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertEvent {
    pub layer: u8,
    pub expert: u8,
    pub selected: bool,
    pub routed_tokens: u16,
    pub packed_bytes: u32,
    pub decoded_bytes: u32,
    pub layer_read_us: u64,
    pub layer_decode_us: u64,
    pub layer_compute_us: u64,
}

/// A complete visualization trace with a deliberately small event ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub model: &'static str,
    pub schedule: &'static str,
    pub events: Vec<ExpertEvent>,
}

/// The selected experts for one token at one transformer layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteEvent {
    pub layer: u8,
    pub token: u16,
    pub experts: [u8; 8],
}

/// A bounded, inert routing trace suitable for checked-in empirical data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingTrace {
    pub model: &'static str,
    pub events: Vec<RouteEvent>,
}

impl RoutingTrace {
    pub const MAX_EVENTS: usize = 24 * 128;

    /// Adds one route while bounding data derived from an untrusted invocation.
    ///
    /// # Errors
    /// Returns an error for an out-of-range expert or after the event bound.
    pub fn push(&mut self, event: RouteEvent) -> Result<(), &'static str> {
        if self.events.len() == Self::MAX_EVENTS {
            return Err("routing trace exceeds 3,072 events");
        }
        if event.experts.iter().any(|expert| *expert >= 32) {
            return Err("routing trace expert is outside 0..32");
        }
        self.events.push(event);
        Ok(())
    }

    /// Renders schema v1 without executable serialization or dependencies.
    #[must_use]
    pub fn to_json(&self) -> String {
        let events = self
            .events
            .iter()
            .map(|event| {
                let experts = event
                    .experts
                    .iter()
                    .map(u8::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "{{\"layer\":{},\"token\":{},\"experts\":[{experts}]}}",
                    event.layer, event.token
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"emufpga.moe-routing.v1\",\"model\":\"{}\",\"events\":[{events}]}}",
            self.model
        )
    }
}

impl Trace {
    pub const MAX_EVENTS: usize = 24 * 32;

    /// Adds an event, refusing an unexpectedly large trace.
    ///
    /// # Errors
    /// Returns an error after the fixed 24-layer by 32-expert bound is full.
    pub fn push(&mut self, event: ExpertEvent) -> Result<(), &'static str> {
        if self.events.len() == Self::MAX_EVENTS {
            return Err("visualization trace exceeds 768 events");
        }
        self.events.push(event);
        Ok(())
    }

    /// Renders schema v1 without a general-purpose serializer or executable data.
    #[must_use]
    pub fn to_json(&self) -> String {
        let events = self
            .events
            .iter()
            .map(event_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"emufpga.serial-moe.v1\",\"model\":\"{}\",\"schedule\":\"{}\",\"timing_scope\":\"measured-layer-total\",\"events\":[{events}]}}",
            self.model, self.schedule
        )
    }
}

fn event_json(event: &ExpertEvent) -> String {
    format!(
        "{{\"layer\":{},\"expert\":{},\"selected\":{},\"routed_tokens\":{},\"packed_bytes\":{},\"decoded_bytes\":{},\"layer_read_us\":{},\"layer_decode_us\":{},\"layer_compute_us\":{}}}",
        event.layer,
        event.expert,
        event.selected,
        event.routed_tokens,
        event.packed_bytes,
        event.decoded_bytes,
        event.layer_read_us,
        event.layer_decode_us,
        event.layer_compute_us
    )
}

/// Pure state transition used by play, pause and step controls.
#[must_use]
pub const fn next_frame(current: usize, event_count: usize, playing: bool) -> usize {
    if playing && event_count > 0 {
        (current + 1) % event_count
    } else {
        current
    }
}
