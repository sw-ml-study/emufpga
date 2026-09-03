# Qwen3 8B Quantized GPU Ladder

Vision: establish a reproducible medium-small LLM baseline on the local RTX 5060 Ti, then compare safe quantized formats and build toward Rust/SPM execution without confusing loader defects, quantization loss, or runtime behavior.

Constraints:
- Keep every individual downloaded model artifact below 12 GB.
- Store weights and generated artifacts outside Git under /disk1/tmp.
- Start at an 8192-token context cap for comfortable 16 GB VRAM headroom.
- Prefer official Qwen artifacts as references; use reputable community imatrix GGUF only for specifically documented experiments.
- Never execute pickle checkpoints in the production path.
- Record exact revisions, hashes, commands, GPU memory, latency, throughput, and output comparisons.

Planned ladder:
1. qwen3-q4-baseline: characterize the installed Ollama Qwen3 8B Q4_K_M model and establish a deterministic GPU baseline.
2. qwen3-q6-reference: acquire the official Q6_K GGUF, verify integrity/metadata, and compare quality, VRAM, and throughput against Q4_K_M.
3. gguf-rust-ingestion: implement bounded Rust GGUF metadata/tensor ingestion with fixtures and parity against trusted tooling.
4. rust-forward-parity: execute a minimal Qwen3 forward path in Rust at fixed prompts and compare logits/tokens against the Ollama or llama.cpp oracle.
5. quant-ladder: compare Q8_0 and IQ4_XS, optionally IQ3_M, using a fixed correctness and performance corpus.
6. fp8-awq-comparison: evaluate official FP8 Safetensors and AWQ only after GGUF correctness is established and runtime support is confirmed.

Decision gates:
- Correctness before optimization.
- Q6_K is the high-fidelity GGUF reference unless Q8_0 reveals material disagreement.
- Do not advance a format whose loader cannot reproduce tensor metadata and deterministic prompt outputs.