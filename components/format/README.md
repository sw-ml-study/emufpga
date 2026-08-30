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
| `spm-codec` | 2-bit ternary packing; Q4 arrives in saga 3 |
| `spm-layout` | op descriptors, column-major tiling, scale groups |
| `spm-walk` | forward-only cursor over the (stream, group) sequence |
| `spm-file` | reader and writer composing the above |

`spm-bytes`, `spm-codec`, `spm-layout` and `spm-walk` are `no_std` and
allocation-free, so Front 3 (RP2350) can use them unchanged.

`SpmReader` exposes no seek, not even privately: step 003 builds
`WeightStream` on top of it and the guarantee has to hold at every
layer.

Built by saga 1 step 2 (spm-format).
