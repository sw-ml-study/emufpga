# llama.cpp SM120 and Rust parity

1. Rebuild and pin llama.cpp with CUDA kernels compatible with the RTX 5060 Ti (compute capability 12.0). Verify the existing Qwen3 8B Q6_K model fully offloads and executes without kernel-image errors. Compare deterministic top logits with the established CPU oracle and document tolerances, memory, and reproducibility.
2. Establish independently observable block-zero intermediate values using a pinned reference implementation or narrowly instrumented llama.cpp build. Compare the Rust single-token embedding and block-zero outputs, correct discrepancies, and record explicit tolerances.
3. Extend the Rust Qwen3 harness to multiple tokens with RoPE, causal grouped-query attention, and KV state. Validate each stage against the reference while preserving bounded GGUF reads.
4. Stream all 36 transformer blocks plus final normalization/output projection and compare final Rust logits to the pinned llama.cpp oracle. Evaluate a maintained Rust CUDA backend only after CPU correctness is established.
