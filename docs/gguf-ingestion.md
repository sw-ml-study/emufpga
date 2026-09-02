# Bounded Rust GGUF ingestion

`spm-gguf` validates GGUF before a later execution backend receives tensor
ranges. It is dependency-free and does not execute model content. Use the
CLI through:

```sh
scripts/inspect-gguf /path/to/model.gguf
```

The reader currently accepts little-endian GGUF v3; scalar metadata and
homogeneous metadata arrays; F32, F16, BF16, Q4/Q5/Q8 legacy blocks; and
Q2_K through Q8_K blocks. It reports absolute, checked tensor byte ranges.
Unsupported versions, metadata types, GGML tensor types, invalid UTF-8,
duplicate keys or tensors, zero dimensions, partial quantization blocks,
unaligned or overlapping tensors, and ranges outside the file are rejected.

Limits are applied before allocation: one million metadata entries, one
million tensors, 16 MiB per string, two million array elements, eight tensor
dimensions, four nested metadata levels, and a 512 MiB total parsed-header
boundary. Arithmetic for positions, element counts, byte lengths, and file
ranges is checked. These limits make malformed inputs fail predictably; they
do not make semantic model content trustworthy.

Candle 0.9.2 was evaluated for reuse. Its GGUF implementation supports
quantized execution, but allocates vectors directly from file-provided
string and array lengths. It is therefore a candidate execution backend only
after this bounded validator, not the trust-boundary parser.

## Official Qwen3 8B Q6_K result

For the pinned, checksum-verified file documented in
`docs/qwen3-8b-q6.md`, the Rust reader reports:

- GGUF v3, 28 metadata entries, 399 tensors.
- Tensor data begins at byte 5,956,384.
- `general.architecture=qwen3`.
- `qwen3.context_length=40960`.
- Tensor types are F32 and Q6_K.

Ollama independently reports Qwen3, 8.2 billion parameters, 40,960-token
context, and Q6_K from the same whole-file SHA-256. Selected raw-range
fingerprints provide stable later-loader gates:

| Tensor | Absolute offset | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `output_norm.weight` | 516,461,344 | 16,384 | `9be52f7d4e9e74c50e9b66fb992c760787abe9a435c550cd6c47e0f168c1267d` |
| `blk.0.attn_q.weight` | 1,044,202,784 | 13,762,560 | `f31b480e4ba25cb0d6b772dc3a185f26ad788405c1c970df84e3100370e331da` |
| `blk.35.ffn_up.weight` | 6,684,611,360 | 41,287,680 | `e67c018fbad1e3fdbe639d85a9a9a14d613a3a8446c70dc95d41c1e318d01e2e` |

The CLI intentionally prints descriptors rather than loading tensor bodies.
Dequantization and execution belong to the next step and must consume only
validated ranges.
