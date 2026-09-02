Wrote the SPM's central claim as two executable MLPL programs, per
docs/research2.txt.

spm_stream.mlpl shows y = W x conventionally, then S = transpose(W) as
the physical consumption view, then a forward-only recursive sweep
that reproduces y exactly -- the punch line being that the arithmetic
did not change, the memory contract did. It then does the same sweep
with +1 add / -1 subtract / 0 bypass and no multiply at all, still
exact, and finally one sweep serving a batch of three matching the
conventional batched matmul.

spm_order.mlpl replays this repo's real TRM layout defect: sorted()
gave alphabetical order while the forward pass wanted qkv, o,
gate_up, down. It counts backward seeks -- zero for execution order,
three for alphabetical.

Every lesson ends in a difference that must be all zero, and every one
of them does. Both files were run before anything was claimed.

Probed the language rather than assuming: found that script mode
echoes only the final value so print() is needed for progressive
output, that UDFs are called with the u: prefix, and that there is no
scalar broadcast from a [1] array. Each of those would have been a
wrong guess.

Ready to move to sw-mlpl/demos unchanged.