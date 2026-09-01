Measure what happens when the parameter store is not RAM.

Every result in docs/results.md was read from a page-cached file on a
64 GB machine. The demanded-bandwidth column is therefore a
REQUIREMENT a store would have to meet, never an observation of one
meeting it. This is the cheapest experiment that could falsify the
central assumption, and it has not been run.

BUILD

A `spm-stream-throttle` crate wrapping any `WeightStream` and limiting
it to a given bytes-per-second, plus a report of how long the engine
spent **waiting on the store**. That stall time is the number: it is
`eta` measured rather than modelled.

Model bandwidth only, and say so. No seek latency, no queueing, no
overlap -- `spm-stream-file` already has no IO overlap, so a
bandwidth-only throttle is consistent with the rest of the stack
rather than a new assumption.

MEASURE

Sweep the serving demo across store speeds spanning real devices:
NVMe, SATA SSD, SAS, spinning disk, 1 GbE. Find where the engine stops
being compute-bound. The prediction from docs/results.md is that at
five clients and bf16 the demand is 258 MB/s, so a store slower than
that should start showing stall time and nothing faster should.

A prediction that survives is worth more than a measurement that had
none, so write it down before running.

REPORT

Stall time, wall clock, and effective tokens per second against store
speed. Say plainly whether the thesis survives: the claim is that a
cheap sequential device can feed this engine, and this is the first
test of it.

A genuinely cold read needs `sudo purge` on macOS, which needs Mike's
password. Note what remains untested.

DISCIPLINE

Hermetic tests, `just check` before committing, no new dependencies --
the stream workspace has none and a rate limiter needs none.
