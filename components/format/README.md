# components/format

The `.spm` container: a physical execution layout, not a model
interchange format. Weights are stored in exactly the order the tensor
engine consumes them, so reading a stream to the end IS the matrix
operation.

Byte layout and the rules three implementations must agree on:
[../../docs/spm-format.md](../../docs/spm-format.md).

| Crate | Responsibility |
| --- | --- |
| `spm-bytes` | little-endian integer primitives |
| `spm-header` | magic, version, endianness, stream count |
| `spm-codec` | 2-bit ternary packing |
| `spm-codec-dense` | dense f32 weights, four bytes each |
| `spm-layout` | op descriptors, column-major tiling, scale groups |
| `spm-walk` | forward-only cursor over the (stream, group) sequence |
| `spm-file` | reader and writer composing the above |

`spm-bytes`, `spm-codec`, `spm-layout` and `spm-walk` are `no_std` and
allocation-free, so Front 3 (RP2350) can use them unchanged.

`SpmReader` exposes no seek, not even privately: step 003 builds
`WeightStream` on top of it and the guarantee has to hold at every
layer.

## Encodings

The `Encoding` discriminant selects both the packing and the byte
length of a group, and readers consult it **per stream**. A file may
mix encodings; a ternary group of four weights is one byte while an
f32 group of four is sixteen.

| code | profile | bytes per group of n |
| ---: | --- | --- |
| 1 | `Ternary2F32I32` | `ceil(n / 4)` |
| 2 | `F32` | `n * 4` |
| 3 | `Bf16` | `n * 2` |
| 4 | `Q6K` | `ceil(n / 256) * 210` |

For `F32`, `Bf16`, and `Q6K` the group scale is inert: the weights carry their own
magnitude, so writers emit 1.0 and readers ignore it.

Built by saga 1 step 2 (spm-format) and saga 2 step 1
(encoding-aware-format).
