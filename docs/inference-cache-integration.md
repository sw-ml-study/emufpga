# Inference cache integration boundary

## Result

A bounded application-owned expert cache now feeds real Gemma-4 Q5_K_M and
Q8_0 arithmetic in the Rust layer oracle. It does not yet feed complete
llama.cpp inference.

With batch four, layer zero selected 20 experts for 32 assignments. Three
accesses per expert exercised cold observation, admit-on-second-use, and a
subsequent cache hit. The adapter recorded 40 hits and 80 misses, retained
96.6 MB within a 128 MiB limit, and had no eviction. Its output matched the
independent packed serial path with maximum error 0.00000381 against the
existing 0.002 tolerance.

The cache owns immutable reference-counted byte buffers. A consumer holds an
`Arc<Vec<u8>>`, so admission or eviction cannot invalidate memory while a
kernel reads it. Model bytes, quantization, routing, and arithmetic are
unchanged.

## Native GGML lease hook

The pinned llama.cpp patch now exposes an experimental expert-span provider at
the CPU `mul_mat_id` operation. Every worker acquires an immutable span for the
selected expert. A barrier ensures every worker has finished its multiply
before any worker releases its lease. Installation is restricted to the period
before graph execution begins; an acquire/release pair must be installed and
removed together.

A counted pass-through provider exercised this hook during complete Gemma-4
inference. It recorded 3,592 acquisitions and 3,592 releases. Resident,
`MADV_DONTNEED`, and pass-through-provider runs produced bit-identical arrays of
262,144 logits, all with SHA-256
`167aaa9c3a8a1a3f7e9e110fde23d4e1b74fc65071a3692b8f0b786896ab2133`.
The pass-through provider returns the original mmap bytes, so this proves hook
placement and balanced lifetime, not cache substitution.

CUDA initialization and `nvidia-smi` both failed because the NVIDIA driver was
unavailable during these runs. Consequently this continuation is a CPU-kernel
correctness result and contains no new VRAM or GPU-placement measurement.

## Why complete llama.cpp is not connected yet

The original CPU MoE operation received one GGML tensor whose `data` pointer
was a stable address in the model mmap. The new hook removes that immediate
pointer obstacle, but no bounded provider is installed by llama.cpp yet.

A correct integration needs one of these explicit runtime contracts:

1. connect the bounded cache to the new callback and make admission/eviction
   concurrency-safe across simultaneous graphs;
2. use a stable reserved virtual-address arena whose pages are populated from the
   backing tier and pinned for the operation; or
3. represent repacked expert tensors as independently owned GGML buffers.

Changing `src0->data` opportunistically is rejected: it risks races and breaks
stride assumptions. Copying into GGML scratch memory for every call is safe but
would obscure the measured cache economics unless separately instrumented.

## What is and is not validated

The adapter proves that cached real quantized bytes can cross a safe ownership
boundary and preserve one-layer arithmetic. Earlier complete inference passed
the executable Rust workload and logit equivalence with mmap-backed expert
reclamation. Combining those facts is not end-to-end validation of the new
cache. Tokens/s, correct tasks/hour, VRAM, and complete-inference physical bytes
remain unmeasured for this adapter.

Data: [`gemma4-inference-cache-adapter.json`](data/gemma4-inference-cache-adapter.json).

## Next step

Implement the first bounded native provider behind an explicit opt-in flag.
Start with a small admit-on-second-use cache, instrument source bytes, hits,
evictions, resident bytes, and wait, then repeat the same exact-logit test. Only
after correctness should the executable c1/c2/c4/c8 workload and GPU placement
be measured. Restore the NVIDIA driver before that measurement.
