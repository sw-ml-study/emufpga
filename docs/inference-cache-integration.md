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

## First bounded native provider

The oracle now includes an explicitly enabled native provider. It makes
64-byte-aligned immutable copies, admits a tensor/expert key on its second
zero-to-active lease transition, uses a configurable byte-bounded LRU, and
never considers an actively leased entry for eviction. A mutex protects entry
state across simultaneous graphs. If disabled, oversized, allocation-limited,
or unable to evict safely, it returns the original mmap span.

The focused test holds one cached entry leased while a competing entry tries to
enter a one-entry cache; the leased bytes remain valid and the competitor falls
back. After release, eviction and admission succeed. Eight threads then share
the immutable entry and finish with balanced leases.

Complete three-step greedy Gemma inference also passes exactly:

| Metric | 1 MiB no-admission control | 128 MiB cache |
| --- | ---: | ---: |
| Final logit SHA-256 | `b4191d...60d9d` | `b4191d...60d9d` |
| Worker acquisitions / releases | 7,432 / 7,432 | 7,432 / 7,432 |
| Worker hits | 0 | 1,830 |
| Loads / evictions | 0 / 0 | 610 / 549 |
| Backing-source bytes | 4,181,264,384 | 4,181,264,384 |
| Peak owned cache | 0 | 134,188,032 bytes |
| Summed callback wait | 12.832 ms | 1,073.685 ms |

The result is correct but not beneficial for one request. The apparent 1,830
hits are primarily other GGML workers sharing a copy loaded for the current
expert operation. The 128 MiB cache turns over before a later token returns to
the same layer, so it saves **zero source bytes** while adding copy and locking
work. Summed callback wait is accumulated across workers and is not wall time.

This rejects a naive claim that a small ordinary LRU helps single-request
inference. The project hypothesis remains narrower: simultaneous agents may
reuse a selected layer/expert while it is already resident, and a layer-aware
policy may retain empirically hot experts across the much longer token reuse
distance. Those cases require the server workload, not this oracle.

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

The bounded provider now proves that copied real quantized bytes can cross the
native lease boundary and preserve complete logits. It does not prove useful
cache economics: its first single-request workload saved no source bytes.
Tokens/s, correct tasks/hour, physical I/O, VRAM, and repeated confidence remain
unmeasured for this provider.

Data: [`gemma4-inference-cache-adapter.json`](data/gemma4-inference-cache-adapter.json).

## Next step

Move the provider into the long-lived server and compare layer-aware admission
against LRU under simultaneous c1/c2/c4/c8 executable agent requests. Count
logical reuse by expert operation rather than worker callbacks and collect
physical I/O. Restore stable NVIDIA access before recording GPU placement,
VRAM, and token-throughput context.
