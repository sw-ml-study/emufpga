# Storage-tier break-even

This experiment measures the 42,212,416-byte all-expert and 10,654,528-byte
selected-union Granite Q6_K block-0 layouts on `large12`. It is deliberately a
host-storage experiment, not evidence of FPGA throughput.

## Result

| source / access | all experts GB/s (p10 / median / p90) | selected GB/s (p10 / median / p90) |
| --- | ---: | ---: |
| rotational HDD, direct-I/O proxy | 0.228 / **0.240** / 0.241 | 0.183 / **0.210** / 0.211 |
| NVMe SSD, direct-I/O proxy | 1.335 / **1.343** / 1.359 | 0.703 / **0.746** / 0.762 |
| rotational HDD, warm cache | 2.341 / **2.417** / 2.526 | 0.971 / **0.978** / 1.009 |
| tmpfs, warm | 1.940 / **1.975** / 1.978 | 0.941 / **0.952** / 1.018 |

Each cell is seven complete reads. `iflag=direct` bypasses the ordinary page
cache but is only a cold-media proxy; it is not a first read after power-on.
The small selected artifact exposes fixed syscall/filesystem costs, so its
GB/s should not be extrapolated as sustained device bandwidth. Device and CPU
provenance, latency, CPU utilization, and every distribution are retained in
[`data/storage-tier-analysis.json`](data/storage-tier-analysis.json).

The immediate conclusion is useful but not glamorous: NVMe is about 5.6 times
the HDD's direct all-expert bandwidth on this host, while cached reads already
outrun the scalar Q6_K work. The current CPU reference remains compute-bound
once weights are cached. Route-aware selected storage cuts bytes by 75%, but
small-read overhead prevents a 4x latency improvement.

## Overlap: measured inputs, unimplemented result

The current Rust `FileWeightStream` has two buffers but refills them
synchronously. It therefore does **not** perform read/decode/compute overlap.
The JSON composes separately measured B1 storage and decode+compute medians to
show an optimistic double-buffer ceiling:

```text
synchronous = storage + decode + compute
ideal overlap = max(storage, decode + compute)
maximum saving = 1 - ideal / synchronous
```

For the all-expert layout, the ceiling is 30.5% on direct HDD and 28.9% on
direct NVMe. For selected experts it is 39.4% on direct HDD but only 15.5% on
direct NVMe. These are **not observed speedups**. They are the maximum that a
future prefetch implementation could recover under a two-stage model, before
threading, DMA, queueing, and chunk-boundary costs. Calling them measured
double-buffer performance would be misleading.

This also explains the break-even physics: overlap can hide only the smaller
of I/O time and work time. A faster store reduces the time available to hide;
a faster tensor engine makes storage more visible. Multiple requests help when
their routed activations reuse an expert during one weight pass, increasing
useful arithmetic per fetched byte rather than merely moving the same bytes
faster.

## Reproduce

Large generated layouts remain outside Git under `/disk1/tmp`. Run:

```sh
scripts/bench-storage-tier /disk1/tmp/storage-tier.csv \
  hdd-all=/disk1/tmp/granite-all-b1-q6.spm \
  hdd-selected=/disk1/tmp/granite-selected-b1-q6.spm \
  nvme-all=/path/on/nvme/granite-all-b1-q6.spm \
  nvme-selected=/path/on/nvme/granite-selected-b1-q6.spm
node scripts/analyze-storage.mjs /disk1/tmp/storage-tier.csv \
  docs/data/storage-tier-analysis.json docs/data/granite-timing.json
```

Do not globally drop caches. The harness primes ordinary reads and uses direct
I/O only where the filesystem supports it.

## Next falsifying experiment

Implement bounded asynchronous prefetch in the Rust stream, retain identical
model outputs, and compare its observed wall clock against both the synchronous
measurement and the ceiling above. Then repeat the portable matrix on the
available Xeon/PMEM/GPU systems. An FPGA is justified only after a sustained
host/link pipeline cannot meet the target and the proposed arithmetic rate is
high enough to consume that stream.
