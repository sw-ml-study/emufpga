# FPGA PSRAM and LUT estimates for serial MoE experts

This is a planning estimate based on the two MoE models measured in this
repository. It is not an FPGA synthesis result. PSRAM capacity can be estimated
from tensor sizes; LUT use requires choosing a quantized datapath and running
the target FPGA tools.

## Models measured so far

| Model | Total experts | Active experts/token/layer | Expert tensor bytes | Expert-only share |
| --- | ---: | ---: | ---: | ---: |
| Granite 3.1 1B-A400M Q6_K | 32 | 8 | 990,904,320 B | 90.1% of 1.10 GB |
| Gemma 4 26B-A4B Q5_K_M | 128 | 8 | 17,287,086,080 B | 89.5% of 19.32 GB |

The remaining roughly 10% is embeddings, attention, normalization, routers,
output heads, and other dense tensors.

## One expert at one layer

This is normally the useful FPGA unit. The engine processes a layer, releases
the expert, and advances to the next layer.

| Model | Packed bytes/expert/layer | Approximate decoded projection |
| --- | ---: | ---: |
| Granite Q6_K | 1.29 MB average | up to about 2 MiB F32 |
| Gemma Q5_K/Q8_0 | 4.50 MB average; 4.83 MB in the measured stream | larger temporary decode if expanded |

For Gemma, one expert across all 30 layers is about 135 MB. That is not the
recommended small-FPGA layout; it is better to stream one layer at a time.
Granite is about 31 MB for one expert across all 24 layers.

## PSRAM capacity for parallel engines

If PSRAM holds complete packed expert spans for one layer:

| Parallel engines | Granite | Gemma |
| ---: | ---: | ---: |
| 1 | ~1.3 MB | ~4.3-4.8 MB |
| 2 | ~2.6 MB | ~9-10 MB |
| 4 | ~5.2 MB | ~18-20 MB |
| 8 | ~10.3 MB | ~36-39 MB |

Ping-pong buffering roughly doubles those figures. Therefore eight Gemma
engines need approximately 72-80 MB of PSRAM for two buffers, while eight
Granite engines need roughly 20 MB.

A 64-Mbit PSRAM device is 8 MB. That is suitable for one Gemma stream, or
several Granite streams, but not eight complete Gemma expert buffers.

An alternative is block streaming. The standalone serial reader measured only
4 KiB of packed parameter-group residency. This greatly reduces PSRAM capacity,
but requires enough external bandwidth and buffering to prevent the MAC engine
from starving.

## What is being offloaded?

There are three different meanings of "N experts":

1. N experts from the current layer are buffered in PSRAM. This is the table
   above and is the usual first FPGA design.
2. One expert is retained across every layer. This needs about 31 MB for
   Granite or 135 MB for Gemma per expert and is usually too large for a small
   PSRAM system.
3. Expert blocks stream from SSD, DRAM, or another memory into a bounded FIFO.
   Then PSRAM may only need tens or hundreds of KiB per engine, but sustained
   bandwidth becomes the limiting resource.

## LUTs and DSPs

Model size does not determine LUT count. LUT/DSP use is driven by:

- quantization format (Q6_K, Q5_K, Q4, or 1.58-bit);
- vector width and MAC lanes per engine;
- scale and zero-point unpacking;
- accumulator width and saturation;
- activation and routing buffers;
- PSRAM burst/control logic; and
- whether multipliers use DSP blocks or LUTs.

A useful first-order model is:

```text
resource cost = fixed control + engines * lanes_per_engine * cost_per_lane
```

So an exact LUT answer requires a target FPGA part, clock, quantized arithmetic
format, and lane count. It would be misleading to invent a number before
synthesis.

The sensible progression is:

- first synthesize one narrow Granite Q6_K engine;
- then synthesize one narrow Gemma Q5/Q8 engine;
- measure PSRAM bandwidth, MAC utilization, and numerical error;
- scale to 2/4/8 engines only if the memory interface remains fed.

1.58-bit arithmetic may reduce both PSRAM bandwidth and multiplier complexity,
but it needs a separate accuracy oracle. It should be treated as a later
datapath experiment, not assumed equivalent to Q6_K or Q5_K.

## Practical recommendation

For a Tang Nano-class first experiment, target one Granite expert engine or one
narrow Gemma engine with a small PSRAM double buffer. Demonstrate sustained
streaming, correct selected-expert output, and no buffer overrun. The important
measurement is not "how many experts fit," but whether the chosen number of
MAC lanes can consume expert bytes at the rate the PSRAM interface supplies
them.
