One sweep of the weights, five clients, five different questions. The
first positive end-to-end result this project has produced.

Five prompts of different lengths (1,2,3,4,6 tokens) greedy-decoded 8
tokens each on the real bf16 SmolLM2-135M. Every token matches
transformers decoding that prompt ALONE, decoded together off one
sweep per step: 212,336,640 weight bytes serving 5 clients = 42.5 MB
per generated token against 212.3 MB for one.

Why nothing before could win: the baseline was the same scalar loop,
the working set fit in RAM every time, there was no decoding at all,
and 'batch' meant positions in one prompt when decoding is batch-1 per
client. Concurrent clients are the only place batch > 1 arises during
generation.

The mechanism, in answer to 'how does it work when they ask different
questions': a matmul applies the same W to each client's own x, so the
activation buffer is clients x hidden and a weight is applied to every
row before being discarded. A question lives only in the client's
activation row, KV cache and position; the sole place clients could
interact is attention, which runs per client and carries no weights.
That is plan.md section 3's asymmetry finally carrying load.

Correctness is the claim: a test asserts five clients batched produce
bit-identical tokens to the same clients alone, and another asserts
weight traffic is independent of client count, measured from
descriptors rather than arithmetic.

No throughput claim -- the engine is still scalar. What amortizes is
traffic, a property of the schedule.

New components/serve workspace. Gate clean across all seven: 222
checks, 0 failed, 1 standing warning. Pushed 144b0a7.