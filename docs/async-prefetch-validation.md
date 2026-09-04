# Asynchronous prefetch validation

The Rust file backend now has a real bounded prefetch implementation. A worker
fills the next 64 KiB chunk while the consumer decodes and applies the current
chunk. A zero-capacity rendezvous channel limits the live payload to two chunks
rather than turning “streaming” into an unbounded read-ahead cache.

## Correctness

Unit tests compare synchronous and prefetched streams across deliberately odd
buffer and destination boundaries, empty input, and rewind. A complete
24-layer Granite 3.1 1B-A400M Q6_K forward pass retained:

- the same final top-1 token, `444`;
- 9/10 top-logit overlap;
- maximum expert and combined absolute error `0.00097656`, below the `0.002`
  gate;
- 1,013,097,984 logical expert bytes, zero mid-region rewinds, and the same
  132,306-byte conservative input-path bound.

## Observed B1 result

Each cell compares seven synchronous and seven prefetched runs of the same
correct expert phase. Positive is faster. `nocache-proxy` applies per-file GNU
`dd iflag=nocache`; it is safe and non-global, but is not a power-on cold read.

| tier | layout | warm | nocache proxy |
| --- | --- | ---: | ---: |
| HDD | all experts | −0.7% | +4.1% |
| HDD | selected union | −0.5% | +2.2% |
| NVMe | all experts | +11.7% | +5.1% |
| NVMe | selected union | −5.5% | +4.1% |

The result is **mixed**, not a general speedup. Cache-bypass cases show a small
2–5% gain, well below step 014's ideal 15–39% ceilings. Warm cases span noise,
one apparent gain, and regressions. The background thread can hide only reads
that finish while useful decode/MAC work occurs; thread scheduling, handoff,
copying, 64 KiB boundaries, and already-fast cache reads consume the margin.
The full distributions and CPU observations are in
[`data/prefetch-analysis.json`](data/prefetch-analysis.json).

The named HDD/NVMe tiers identify artifact placement, but warm reads primarily
measure page cache. The per-file cache-bypass proxy is closer to the underlying
device without claiming strict cold-media control. A production test should
use direct aligned I/O or `io_uring`, a prebuilt immutable artifact, and device
telemetry.

## Meaning for the project

Prefetch is necessary plumbing, not the value proposition. It preserves the
bounded stream contract and sometimes hides a few milliseconds, but the scalar
CPU expert work still dominates many cases. The project succeeds only if a
small/old GPU plus serial expert supply runs an oversized MoE at a useful rate
faster than ordinary CPU/System-RAM offload. See the predeclared deployment
criteria in [`claim-scorecard.md`](claim-scorecard.md).

Reproduce the matrix after preparing matching canonical artifacts:

```sh
scripts/measure-prefetch /disk1/tmp/prefetch.csv \
  hdd:/path/hdd/all.spm:/path/hdd/selected.spm \
  nvme:/path/nvme/all.spm:/path/nvme/selected.spm
node scripts/analyze-prefetch.mjs /disk1/tmp/prefetch.csv \
  docs/data/prefetch-analysis.json
```
