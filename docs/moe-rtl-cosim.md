# Serial MoE hardware-shaped validation

`q6-pipeline` is the first hardware-shaped model of the proposed accelerator,
not a performance claim. It accepts exact 210-byte GGML `Q6_K` blocks and
models storage fetch, a bounded byte FIFO, decoder lanes, MAC lanes, selection
gating, starvation, and backpressure. Each block records issue, decode-done,
accumulate-done, and FIFO-level events for visualization.

The decoder is independent of `spm-gguf`; deterministic tests compare every
decoded `f32` bit against that established oracle. Selected and unselected
passes consume identical bytes, while unselected passes produce no values and
spend zero MAC cycles. Slow-fetch and fast-fetch tests preserve identical
answers while moving the bottleneck from starvation to backpressure.

Raw Q6_K has no checksum, so arbitrary corruption can decode into a plausible
wrong number. The model supports an outer transport checksum and tests that a
changed byte fails loudly. Truncated blocks always fail without framing.

## What cycles mean

The model reports exact results under explicit abstract widths. A cycle is a
unit of ordering and resource demand, not elapsed time. Converting cycles to
tokens/second requires both an achieved post-route clock and a measured memory
interface. Until synthesis and place-and-route provide those, cycle projections
must remain separate from host measurements.

The committed known-answer report uses a 420-byte FIFO, 16 fetch bytes/cycle,
eight decoder lanes, and 16 MAC lanes:

| block mode | total cycles | fetch | decode | MAC | initial starvation |
|---|---:|---:|---:|---:|---:|
| unselected drain | 46 | 14 | 32 | 0 | 14 |
| selected compute | 62 | 14 | 32 | 16 | 14 |

Run `cargo run -p q6-pipeline --example report` through `scripts/serial.sh` to
reproduce it. The cycle components overlap, so their sum is not generally the
total for multi-block streams.

## Validation ladder

1. **Done:** exact Q6_K decoder parity and bounded failure behavior.
2. **Done:** FIFO starvation/backpressure and selected MAC gating semantics.
3. **Next:** replay real `.spm` expert regions and compare matrix partial sums.
4. **Next:** express the same state machine in synthesizable RTL and co-simulate.
5. **Later:** synthesize/place/route for a named device and measure its memory.
