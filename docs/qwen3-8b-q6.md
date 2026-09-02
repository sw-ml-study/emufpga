# Qwen3 8B Q6_K reference

The high-fidelity GGUF reference is the official Qwen artifact:

- Repository: `Qwen/Qwen3-8B-GGUF`.
- Revision: `7c41481f57cb95916b40956ab2f0b139b296d974`.
- File: `Qwen3-8B-Q6_K.gguf`.
- Size: 6,725,899,040 bytes.
- SHA-256:
  `cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6`.
- Local path: `/disk1/tmp/qwen3-8b-q6/Qwen3-8B-Q6_K.gguf`.

The pinned download URL is:

```text
https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q6_K.gguf
```

`scripts/import-qwen3-q6-ollama` checks the exact size and SHA-256 before
registration. Ollama identifies the preserved blob by the same SHA-256 and
reports Qwen3 architecture, 8.2 billion parameters, 40,960-token context,
4,096-element embeddings, and Q6_K quantization.

## Controlled wrapper

A bare GGUF import in the installed Ollama version produced only
`TEMPLATE {{ .Prompt }}`. That changed the benchmark prompt from 40 tokens
to 24 and made the initial result invalid as a quantization comparison. The
import script therefore copies the installed Q4_K_M model's template,
sampling defaults, stop tokens, and license, replacing only its `FROM`
weight layer with the checksum-verified Q6_K file. It does not requantize or
modify the GGUF.

Register and benchmark it with:

```sh
scripts/import-qwen3-q6-ollama
QWEN_MODEL=qwen3:8b-q6 scripts/bench-qwen3-ollama
```

## Q4_K_M versus Q6_K

Both measurements used the same 8,192-token context, prompt, Qwen chat
wrapper, disabled thinking, temperature zero, seed 42, one warmup, and three
measured runs on the RTX 5060 Ti.

| Measurement | Q4_K_M | Q6_K |
| --- | ---: | ---: |
| Stored size reported by Ollama | 5.2 GB | 6.7 GB |
| Ollama GPU placement | 100% | 100% |
| Peak whole-device VRAM | 6,804 MiB | 8,080 MiB |
| Mean generation throughput | 20.83 token/s | 16.64 token/s |
| Generated tokens before stop | 120 | 110 |
| Prompt tokens | 40 | 40 |

Q6_K used 1,276 MiB more observed GPU memory and generated about 20% more
slowly. Request-duration means are not directly comparable because the two
quants reached their stop token after different output lengths. Each quant
was internally deterministic across its three runs, but their response text
differed, as expected when quantization changes logits near token-choice
boundaries.

There is one remaining provenance caveat: the installed Q4_K_M weight blob
has SHA-256 `a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`,
whereas the official Qwen repository's pinned Q4_K_M artifact advertises
SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
The installed blob is therefore not byte-identical to that official GGUF.
This benchmark is a valid comparison of the local deployment choices, but
not a strictly controlled study of quantization bits alone.

The aligned Q6 response SHA-256 was
`2f9afb31924945afc7f9c06e0b989197d0e7307db250c753181e89ae9a7e9ad9`.
Raw aligned results remain at
`/disk1/tmp/emufpga-qwen3-q6-aligned-20260902`.

## Decision

Q6_K remains the high-fidelity GGUF reference for Rust ingestion: it fits
comfortably, remains fully GPU-resident, and exercises richer GGUF block
types than a single Q4 baseline. Q8_0 would still be useful later as a
quality-ceiling point because this one-prompt test cannot establish whether
Q6_K is close enough to unquantized logits. A controlled quantization study
should first use the official Q4_K_M from the same pinned repository
revision. Neither additional download is needed before the Rust loader
proves exact metadata and tensor-byte parity on Q6_K.
