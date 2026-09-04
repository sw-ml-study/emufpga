# MoE memory, streams, and economics

This note describes the pinned Granite 3.1 1B-A400M Q6_K experiment. Values
are binary sizes unless a measurement is explicitly reported as decimal
bytes.

## ELI5 picture

Imagine a school with 32 specialist teachers. Each question needs eight of
them. The router is the receptionist who chooses the eight.

Keeping every teacher in the classroom is fast but needs a large classroom.
Keeping one teacher's notebook at a time needs little desk space, but someone
must carry notebooks past the desk. If the notebooks are stored as one long
tape, there are two choices:

- make a special tape containing only today's eight teachers, which is small
  but requires knowing the receptionist's answer before making the tape; or
- put all 32 teachers on the tape, read it from beginning to end, and ignore
  24 notebooks. This works for any question but moves four times as much
  expert data.

Batching is asking several children's questions while each notebook is open.
The notebook is carried once and reused. That is where serial MoE can recover
much of its speed.

## Model and sequential streams

The GGUF artifact is 1,099,212,096 bytes (1.024 GiB). It contains about 1.33
billion parameters, but only about 400 million participate in one token. The
entire file fits comfortably in the RTX 5060 Ti's approximately 15 GiB usable
VRAM, so this model is a correctness target rather than a capacity challenge.

One layer contains these source streams:

| weights | complete Q6_K/F32 bytes | active-token bytes |
| --- | ---: | ---: |
| Q, K, V, attention output | 2,580,480 | 2,580,480 |
| router | 131,072 | 131,072 |
| 32 experts, up/gate/down | 41,287,680 | 10,321,920 for eight |
| two F32 norms | 8,192 | 8,192 |
| layer total | 44,007,424 | 13,041,664 |

Across 24 layers, the transformer weights account for 1,056,178,176 source
bytes. Embeddings, output norm, metadata, and alignment bring the artifact to
the measured 1,099,212,096 bytes.

The `.spm` proof currently uses F32 because `.spm` has no Q6_K execution
profile. Its real block-0 measurements are:

| layout | streams | file bytes | useful streamed bytes | rewinds |
| --- | ---: | ---: | ---: | ---: |
| known-route, eight experts | 25 | 50,549,568 | 50,548,736 | 0 |
| dynamic, all 32 experts | 97 | 201,792,576 | 50,548,736 | 0 |

The dynamic stream therefore uses 25.05% of the bytes it reads and passes
about 151.24 MB without arithmetic. This is not hidden as "sparse" speedup:
MoE saves multiplication, but a single forward-only tape does not save expert
traffic unless routing can select storage before the scan.

The stream reader reports a 4,096-byte resident parameter group. The file
backend additionally owns two 64 KiB I/O buffers, so the honest streaming
input footprint is about 132 KiB plus small metadata, activations, and
accumulators. The selected-only parameter residency ratio is 0.0000810; the
all-expert ratio is 0.0000203.

## KV cache and context

The KV cache remembers attention keys and values for earlier tokens. Granite
has 24 layers, eight KV heads, and 64 lanes per head. With F16 cache entries:

```text
bytes per token
  = 24 layers x 8 heads x 64 lanes x 2 (K and V) x 2 bytes
  = 49,152 bytes
  = 48 KiB
```

| context tokens | F16 KV cache | F32 KV cache |
| ---: | ---: | ---: |
| 32 | 1.5 MiB | 3 MiB |
| 1,024 | 48 MiB | 96 MiB |
| 4,096 | 192 MiB | 384 MiB |
| 8,192 | 384 MiB | 768 MiB |
| 16,384 | 768 MiB | 1.5 GiB |
| 32,768 | 1.5 GiB | 3 GiB |
| 65,536 | 3 GiB | 6 GiB |
| 131,072 advertised maximum | 6 GiB | 12 GiB |

llama.cpp measured exactly 1.5 MiB for its 32-token F16 allocation. The Rust
fixture evaluates an already supplied prompt in one pass; it does not yet
retain a production autoregressive KV cache. For its five-token prompt, the
current layer's F32 K and V vectors occupy only 20 KiB. A real Rust generation
loop must budget the all-layer table above.

The advertised 128K context is therefore not free. Model plus F16 KV is about
7 GiB before CUDA workspaces, output buffers, allocator slack, and batching.
It should fit in 16 GB for this model, but maximum context and maximum batch
cannot both be assumed without measurement.

## Other live memory

For one F32 token:

| live value | size |
| --- | ---: |
| hidden or residual vector, width 1,024 | 4 KiB |
| query vector | 4 KiB |
| key vector, width 512 | 2 KiB |
| value vector, width 512 | 2 KiB |
| one expert up/gate/activation, width 512 | 2 KiB each |
| one expert output/accumulator, width 1,024 | 4 KiB |
| one decoded F32 expert projection | 2 MiB |

These scale roughly linearly with batch tokens, except a decoded expert
projection is shared across the batch. This is why parameters and KV cache,
not single-token activations, dominate this small model. The measured scalar
GGUF run sampled 8,256 KiB peak process RSS because it mmaped the model and
touched bounded ranges; that sample is not GPU memory and not a kernel
high-water mark.

## Batching math

Let `N = 32` experts, `k = 8` selected experts, and `B` tokens.

- Separate selected-expert scans need `B x k` expert scans.
- One full all-expert sweep needs `N` scans.
- They break even when `B x k = N`, so `B = 32 / 8 = 4` tokens.

For the measured five-token sentence there are 40 token/expert assignments
but only 24 unique selected experts:

| schedule | expert scans | useful assignments per scan |
| --- | ---: | ---: |
| separate per token | 40 | 1.00 |
| group by selected expert | 24 | 1.67 |
| immutable all-expert sweep | 32 | 1.25 |

The compact logical routing state is 40 bytes per token: eight one-byte IDs
and eight F32 scores. The current general Rust representation uses `usize`
IDs, so its vector payload is 96 bytes per token, or 480 bytes for five,
before `Vec` bookkeeping. Both are tiny beside one 2 MiB decoded projection.

Four tokens is only the traffic break-even point. Real break-even also depends
on storage startup latency, how evenly routes overlap, activation buffering,
and whether the compute unit can use all batched work without stalling.

## Physics and economics

Physics supplies two different limits:

1. **Capacity:** how many bytes fit near the compute unit. Serial execution
   wins here because it needs one small group or projection, not all weights.
2. **Bandwidth:** how many bytes can cross the wire each second. Reading unused
   experts still consumes time and energy even when no multiplication occurs.

Roughly, lower-bound layer time is:

```text
time >= max(bytes transferred / storage bandwidth,
            arithmetic operations / compute throughput)
```

The slower side wins. MoE reduces the arithmetic term from 32 experts to
eight. A blind linear sweep leaves the expert-byte term at 32. Batching raises
the arithmetic done per byte and is therefore the key economic lever.

GPU VRAM is expensive but very wide and close to the arithmetic units. Host
RAM is cheaper and slower; NVMe capacity is cheaper again and much slower.
Serial streaming trades expensive capacity for bandwidth demand. That can be
a good bargain when weights otherwise do not fit, or when batching reuses
each arriving weight enough times. It is a poor bargain when 75% of the tape
is discarded and latency matters.

The measured 3.4-second dynamic run includes creating the 192.4 MiB F32 file
and then reading a warm local file. It is a correctness measurement, not an
SSD, GPU, FPGA, energy, or cold-cache throughput benchmark. Those require
separate counters and controlled storage conditions.
