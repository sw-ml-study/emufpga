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
