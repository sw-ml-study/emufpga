Took up the sw-mlpl changes. Verified all four asks against the local
build rather than the changelog: at(v,i) (which also indexes matrix
rows), length-1 broadcast, infix comparisons, dataflow.

Simplified both existing lessons. The predicted payoff arrived exactly
as written -- the sweep's heart went from two reshape/gather_rows
nests to 'row = at(S, i); a = at(x, i)', which is one activation
meeting one row of weights. Every difference still prints zero.

Wrote the two lessons that dataflow unblocked. spm_residency.mlpl:
Rp = 1 versus 1.52e-5, and the memory hierarchy drawn both ways so the
contrast reads as weights moving to a different KIND of memory rather
than just shrinking; ends on the KV cache, 300 GB of model against
1.44 GB of RAM. spm_pipeline.mlpl: the measured store sweep from step
12, and it re-derives max(compute, bytes/rate) IN the language,
printing residuals of 0.02-0.22 s against the real run -- a model
checked in the teaching material rather than asserted in prose.

UPSTREAM.md records all four as shipped and adds two asks that only
became visible once the renderer existed: an honest width channel for
a 65,000:1 ratio, and a back-edge that reads as a rewind.