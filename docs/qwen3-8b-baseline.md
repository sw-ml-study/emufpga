# Qwen3 8B quantized GPU baseline

This experiment uses the locally installed `qwen3:8b` as the first
medium-small LLM rung. Ollama reports Qwen3 architecture, 8.2 billion
parameters, a 40,960-token model context, and Q4_K_M quantization. Its
immutable model blob is
`sha256:a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`.
The weight blob is not copied into Git.

Run the baseline with:

```sh
scripts/bench-qwen3-ollama
```

Results default to a timestamped directory under `/disk1/tmp`. The harness
uses an 8,192-token context cap, disables thinking, sets temperature to zero
and seed to 42, performs one warmup, and then measures three identical runs.
Override these with `QWEN_MODEL`, `QWEN_RUNS`, `QWEN_NUM_CTX`, and
`QWEN_NUM_PREDICT`.

The output includes Ollama's prompt and generation timings, GPU samples,
reported model placement, and the full generated responses. Generation is
made as repeatable as Ollama permits, but exact token identity is not a
portable promise across Ollama or CUDA versions, kernels, quantizers, and
hardware. Treat the saved response as an observation. Loader identity,
tensor metadata, logits within a stated tolerance, and aggregate performance
are better cross-runtime gates.

The warmup separates the large cold model-load cost from steady-state prompt
evaluation and token generation. Ollama retains the model for five minutes;
later runs therefore measure a warm server and may benefit from internal
caches. The GPU sampler observes whole-device memory rather than attributing
every MiB to Ollama, so run this without unrelated GPU workloads.

## Recorded local result

The committed result is intentionally concise. The detailed responses and
samples remain in `/disk1/tmp`.

- GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB, driver 610.57.04.
- Model: Qwen3 8.2B Q4_K_M, fully GPU-resident according to `ollama ps`.
- Context cap: 8,192 tokens.
- Ollama placement: 100% GPU with an 8,192-token allocated context.
- Observed peak whole-device memory: 6,804 MiB.
- Three warm generation runs: 120 tokens each at 19.24, 19.59, and
  23.67 tokens/s; mean 20.83 tokens/s.
- Mean request duration: 6,140 ms. Warm per-request load bookkeeping was
  142-182 ms; the separate cold probe took about 15.1 seconds to load.
- All three generated responses were byte-identical. Their response-text
  SHA-256 is
  `c573523a0fbc4493033721a77e34e349c3e4e9921a53e96f78fc73e44d81b64d`.

These figures are observations from 2026-09-02, not performance guarantees.
Prompt-evaluation throughput varied substantially, consistent with caching
and the prompt's small 40-token size, so it is not yet a useful comparison
metric. The raw run is retained locally at
`/disk1/tmp/emufpga-qwen3-q4-baseline-20260902`.

The next gate is the official Q6_K GGUF. Compare it with this same prompt and
harness settings before implementing GGUF ingestion in Rust.
