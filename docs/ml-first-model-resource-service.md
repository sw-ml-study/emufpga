# ML-first model resource service

## Result

A minimal Rust user-space service now treats an expert stream as a verified,
content-addressed resource and serves multiple Unix-socket clients from one
bounded forward sweep. It uses JSON plus SHA-256, not pickle or another
executable serialization format. Immutable model, tensor, and expert-stream
resources are distinct from mutable per-request KV-cache and activation
resources.

The first measurement is useful but narrower than inference. It replays the
validated 169,688,672-byte Gemma-4 layer-0 expert artifact and hashes every
byte. Three cold runs of each policy were made at 1, 2, 4, and 8 clients.

| clients | independent user bytes | shared user bytes | physical HDD bytes, either | independent p50 | shared p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 169.69 MB | 169.69 MB | 169.69 MB | 949.6 ms | 954.5 ms |
| 2 | 339.38 MB | 169.69 MB | 169.69 MB | 964.3 ms | 942.0 ms |
| 4 | 678.75 MB | 169.69 MB | 169.69 MB | 957.3 ms | 965.0 ms |
| 8 | 1,357.51 MB | 169.69 MB | 169.69 MB | 1,263.2 ms | 961.8 ms |

At eight clients, the shared scheduler removes 8x duplicate user-space stream
traversal and bounds buffers to 2 MiB instead of 8 MiB. Its median completes
24% sooner in this hash workload. This is not an 8x storage saving: Linux
already coalesces simultaneous independent readers through the shared page
cache, so both policies issue one artifact-sized physical HDD read. That is an
important warning against counting logical reads as physical I/O.

## Resource contract

An `emufpga.resource.v1` manifest records a SHA-256 content identity, exact
byte length, resource kind, storage tier, mutability, and residency limit.
Model, tensor, and expert-stream resources must be immutable. KV and
activation resources must be mutable. The service rejects a mismatched hash,
schema, identity, size, or mutability contract before serving bytes. See the
checked [Gemma manifest](data/gemma4-layer0-expert-resource.json).

## Scheduler and IPC

Clients send `GET sha256:<digest>` over a Unix-domain socket. `serve-once`
collects the declared client count, requires an identical resource ID, performs
one `PrefetchFileWeightStream` sweep, and returns the verified byte count and
digest to every client. The stream API cannot seek. Its two asynchronous
buffers have a fixed combined 2 MiB residency in the measured configuration.

This one-batch protocol deliberately omits authentication, cancellation,
backpressure, deadlines, and long-lived admission. It establishes the process
boundary and shared-sweep behavior without pretending to be production RPC.

## Possible future OS primitives

Only after the user-space service demonstrates value would these kernel-facing
interfaces be justified:

- immutable content-addressed extents with tier and sequential-order hints;
- a bounded-residency lease independent of the ordinary page cache;
- queues keyed by resource or expert ID rather than only process or thread;
- NUMA-aware placement of streams, activations, and compute;
- accounting for useful applications per fetched byte across processes;
- deadlines and fairness for clients waiting on the same passing expert.

Linux already supplies files, page cache, Unix sockets, threads, cgroups,
NUMA policy, and asynchronous reads. The experiment must first show which
missing semantic materially improves capacity or completed work.

## Not yet established

- No expert arithmetic, GPU handoff, KV cache, or coding task is performed by
  this byte replay, so it says nothing new about output quality.
- Cgroup limits, NUMA placement, queue depth, wall energy, cancellation, and
  sustained fairness are not measured.
- Linux already shares physical page-cache reads. Prospective value lies in
  coordinated expert computation and bounded residency, not merely disk I/O.
- The next experiment must route real coding workloads and measure whether one
  expert application can serve multiple waiting activations.

Raw aggregates are in
[`data/resource-service-analysis.json`](data/resource-service-analysis.json).
Reproduce them with `scripts/measure-resource-service` and
`scripts/analyze-resource-service.mjs`.
