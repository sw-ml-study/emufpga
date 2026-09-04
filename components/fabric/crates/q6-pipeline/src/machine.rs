use crate::decode::decode_block;
use crate::{BLOCK_BYTES, BLOCK_VALUES, BlockEvent, Config, Cycles};

pub(crate) struct State {
    pub values: Vec<f32>,
    pub events: Vec<BlockEvent>,
    pub cycles: Cycles,
    pub fetched: usize,
    fifo: usize,
    consumed: usize,
    busy: usize,
}
impl State {
    pub fn new(blocks: usize, selected: bool) -> Self {
        Self {
            values: Vec::with_capacity(if selected { blocks * BLOCK_VALUES } else { 0 }),
            events: Vec::with_capacity(blocks),
            cycles: Cycles::default(),
            fetched: 0,
            fifo: 0,
            consumed: 0,
            busy: 0,
        }
    }
    fn issue(&mut self, bytes: &[u8], selected: bool, config: Config) {
        self.fifo -= BLOCK_BYTES;
        if selected {
            decode_block(
                &bytes[self.consumed * BLOCK_BYTES..(self.consumed + 1) * BLOCK_BYTES],
                &mut self.values,
            );
        }
        let decode = BLOCK_VALUES.div_ceil(config.decoder_lanes);
        let mac = if selected {
            BLOCK_VALUES.div_ceil(config.mac_lanes)
        } else {
            0
        };
        self.cycles.decode += decode as u64;
        self.cycles.mac += mac as u64;
        self.busy = decode + mac;
        self.events.push(BlockEvent {
            block: self.consumed,
            issued: self.cycles.total,
            decoded: self.cycles.total + decode as u64,
            accumulated: self.cycles.total + (decode + mac) as u64,
            fifo_after_issue: self.fifo,
        });
        self.consumed += 1;
    }
    fn fetch(&mut self, len: usize, config: Config) {
        if self.fetched < len {
            let amount = config.fetch_bytes_per_cycle.min(len - self.fetched);
            if self.fifo + amount <= config.fifo_bytes {
                self.fifo += amount;
                self.fetched += amount;
                self.cycles.fetch += 1;
            } else {
                self.cycles.backpressured += 1;
            }
        }
    }
    fn tick(&mut self, blocks: usize) {
        if self.busy > 0 {
            self.busy -= 1;
        } else if self.consumed < blocks {
            self.cycles.starved += 1;
        }
        self.cycles.peak_fifo_bytes = self.cycles.peak_fifo_bytes.max(self.fifo);
        self.cycles.total += 1;
    }
}
pub(crate) fn drive(bytes: &[u8], selected: bool, config: Config, mut state: State) -> State {
    let blocks = bytes.len() / BLOCK_BYTES;
    while state.consumed < blocks || state.busy > 0 {
        if state.busy == 0 && state.consumed < blocks && state.fifo >= BLOCK_BYTES {
            state.issue(bytes, selected, config);
        }
        state.fetch(bytes.len(), config);
        state.tick(blocks);
    }
    state
}
