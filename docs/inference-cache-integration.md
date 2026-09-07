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

## Why complete llama.cpp is not connected yet

The current CPU MoE operation receives one GGML tensor whose `data` pointer is
a stable address in the model mmap. It derives expert addresses as
`src0->data + expert * stride`, and several worker threads use those addresses
inside a graph execution. The existing reclamation patch can advise the kernel
about those mmap pages, but an external cache cannot safely substitute a
different movable buffer per expert.

A correct integration needs one of these explicit runtime contracts:

1. a `mul_mat_id` backend callback that acquires an immutable expert span and
   holds its lease through the worker barrier;
2. a stable reserved virtual-address arena whose pages are populated from the
   backing tier and pinned for the operation; or
3. repacked expert tensors represented as independently owned GGML buffers.

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

Prototype the lease callback at the GGML CPU `mul_mat_id` boundary, initially
behind a feature flag and for one expert tensor type. Validate thread lifetime
and identical logits before measuring performance. If upstream structure makes
that invasive, the stable virtual-address arena is the next candidate.
