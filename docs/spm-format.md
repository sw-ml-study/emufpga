# The `.spm` format

A **physical execution layout**, not a model interchange format.
Weights are stored in exactly the order the tensor engine consumes
them, so opening a stream and reading it to the end IS the matrix
operation.

Three implementations will read this format -- this repository, an
RP2350 streamer, and an FPGA loader -- so the byte layout is a
contract. `components/format/crates/spm-file/tests/golden/tiny.spm`
pins it, and that fixture was written by hand from this document
rather than produced by the writer, so the test compares two
independent readings of the specification.

Version 1.0. All integers little-endian regardless of host.

## File structure

```text
[ header       32 bytes                 ]
[ descriptor   32 bytes ] x stream_count
[ payload                               ]
```

There is deliberately **no offset table**. A sequential reader has no
use for one, and an offset field is an invitation to seek. Each
section follows the previous one immediately.

## Header (32 bytes)

| offset | size | field |
|--------|------|-------|
| 0      | 8    | magic `89 53 50 4D 0D 0A 1A 0A` |
| 8      | 2    | `version_major` |
| 10     | 2    | `version_minor` |
| 12     | 1    | `endianness` (0 = little; the only value defined) |
| 13     | 3    | reserved, zero |
| 16     | 4    | `stream_count` |
| 20     | 12   | reserved, zero |

The magic follows the PNG convention. The high bit of byte 0 catches a
transport that strips to 7 bits; `\r\n` catches CRLF translation;
`\x1a` stops a DOS `type`; the trailing `\n` catches the reverse
conversion. A reader given a larger `version_major` MUST refuse the
file: misparsing a weight stream yields plausible numbers rather than
an obvious error.

## Operation descriptor (32 bytes, one per stream)

| offset | size | field |
|--------|------|-------|
| 0      | 4    | `rows` (M, outputs) |
| 4      | 4    | `cols` (N, inputs) |
| 8      | 4    | `group_size` (G, weights per scale group) |
| 12     | 1    | `encoding` profile |
| 13     | 1    | reserved, zero |
| 14     | 2    | `lane_count` |
| 16     | 16   | reserved, zero |

`encoding` names the weight encoding, scale type and accumulator width
as ONE value rather than three orthogonal fields, because in practice
they co-vary: an FPGA is built for one combination, not for a matrix of
independent choices.

| code | profile | bytes per group of `n` |
|------|---------|------------------------|
| 0    | reserved | -- |
| 1    | `Ternary2F32I32` -- 2-bit ternary weights, `f32` group scales, `i32` accumulators | `ceil(n / 4)` |
| 2    | `F32` -- dense `f32` weights, four bytes each, no packing | `n * 4` |
| 3    | `Bf16` -- dense `bfloat16` weights | `n * 2` |
| 4    | `Q6K` -- one GGML Q6_K block | `ceil(n / 256) * 210` |

**The payload length of a group depends on its stream's encoding**, and
readers must consult it per stream rather than per file. A file may
mix encodings: a ternary group of four weights is one byte while an
f32 group of four is sixteen, and a reader that applied one rule to
the whole file would misalign on the second stream and return garbage
rather than an error.

For the `F32`, `Bf16`, and `Q6K` profiles the group scale is **inert**. The weights carry
their own magnitude, so writers emit `1.0` and readers ignore it. The
field is not removed for this profile on purpose: the group structure
is what makes the stream self-describing, and an encoding that skipped
it would need a reader of its own.

`group_size` must be at least 1. A reader MUST reject an unknown
encoding rather than ignoring it.

## Consumption order

For `y = Wx`, weights are stored **column-major**. Stream position `k`
holds:

```text
W[k % rows][k / rows]
```

Consecutive positions walk DOWN a column, so the engine holds one
activation `x[j]` resident while an entire column of weights streams
past, accumulating into `rows` accumulators. That is the whole reason
the layout exists -- see docs/plan.md section 2.

`Q6K` is the explicit exception: blocks retain GGML source row-major order so
they can be copied without requantization. Its consumer maps decoded position
`k` to `W[k / cols][k % cols]`.

## Payload

Per stream in directory order, per scale group in stream order:

```text
[ scale           f32 little-endian, 4 bytes ]
[ packed weights  ceil(count / 4) bytes      ]
```

The scale is written immediately BEFORE the weights it applies to, so
the engine never seeks to fetch one; it arrives just in time.

A scale applies to `group_size` consecutive weights in stream order.
The final group of a stream is short when `group_size` does not divide
`rows * cols` evenly. Setting `group_size == rows` gives one scale per
column, which lets the engine pre-scale the activation once and keep
the inner loop free of multipliers.

Each group starts on a byte boundary. Alignment wastes a few bits per
group when `count` is not a multiple of four, and buys a hardware
decoder that never straddles a byte boundary. Padding bits in a short
final byte are zero.

A stream declaring zero weights contributes no groups at all, and
readers skip it.

## Ternary weight encoding

Two bits per weight, four weights per byte, **least significant pair
first**: weight `k` of a group occupies bits `2k` and `2k+1` of byte
`k / 4`. LSB-first because a bit-serial consumer -- an FPGA shift
register, or an RP2350 PIO state machine -- receives the low bit first.

| code | bit 1 | bit 0 | meaning |
|------|-------|-------|---------|
| `00` | 0     | 0     | `0`     |
| `01` | 0     | 1     | `+1`    |
| `11` | 1     | 1     | `-1`    |
| `10` | 1     | 0     | invalid |

Bit 0 is **nonzero**: it gates the accumulator enable. Bit 1 is
**negative**: it selects subtract over add. A weight arriving off the
stream therefore drives the arithmetic unit directly, with no decode
stage in between -- the storage stream IS the instruction stream, which
is the point the research makes in its section on ternary weights.

Code `10` would mean "negative zero", which this encoding never
produces. It is left permanently invalid rather than assigned a
meaning, so the hardware decoder stays combinational and stateless.
Readers MUST reject it; it is the cheapest corruption check the format
has.

## Worked example

The golden fixture: a 3x2 matrix (6 weights), `group_size` 4, so one
full group of four and a short final group of two.

```text
offset  bytes                     meaning
0       89 53 50 4d 0d 0a 1a 0a   magic
8       01 00                     version_major = 1
10      00 00                     version_minor = 0
12      00                        endianness = little
13      00 00 00                  reserved
16      01 00 00 00               stream_count = 1
20      00 x12                    reserved
32      03 00 00 00               rows = 3
36      02 00 00 00               cols = 2
40      04 00 00 00               group_size = 4
44      01                        encoding = Ternary2F32I32
45      00                        reserved
46      01 00                     lane_count = 1
48      00 x16                    reserved
64      00 00 80 3f               scale = 1.0
68      71                        +1, 0, -1, +1
69      00 00 00 3f               scale = 0.5
73      03                        -1, 0 (padding bits zero)
```

Total: 74 bytes.

## Extending versus changing

Adding an encoding profile is an **extension**: a new discriminant
value, no bump to `version_major`, and every existing file still reads
byte for byte. The `F32` profile was added exactly this way, and the
ternary golden fixture below was not regenerated.

A reader that meets an unknown discriminant refuses the file, so an
old build reading a new file fails loudly rather than misparsing it.
That is what makes extension safe.

## Changing this format

A layout change is a deliberate act, not a refactor. Bump
`version_major` in `spm-header`, regenerate the golden fixture, and say
so explicitly in the commit message. Three implementations read these
bytes; a silent change makes them disagree without anyone noticing.
