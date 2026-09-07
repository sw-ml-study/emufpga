# Demand-cache break-even

Broad preload lost the short cold test. A persistent cache that admits only
after actual demand does better, but its benefit on `large12` is userspace work,
not disk traffic.

The experiment repeats the same held-out c4 routing window for 12 successive
agent batches without restarting the service. This is deliberately favorable
to reuse and therefore an upper bound. The model file starts cold. Demand-only,
admit-on-second-use, and recency policies read identical Gemma-4 Q5_K_M expert
bytes under 128 and 512 MiB limits.

By batch 12, both cache policies reached an 10.9% hit rate and avoided 1.70 GB,
or 11.6%, of model-file copying. Admit-on-second-use broke even in cumulative
elapsed time by batch two and finished 8.9% sooner at 128 MiB. Expert-read wait
fell 12.6%. The 512 MiB cache produced no additional hits, used roughly 437 MiB
more peak RSS than demand-only, and was slightly less effective in elapsed
time. For this window, 128 MiB dominates 512 MiB.

All policies caused the same 1.178 GB of physical reads. Linux cached the file
after batch one, including for demand-only. Therefore the application cache has
no measured HDD-byte advantage on this ample-RAM host. Its value is fewer
copies and less userspace wait. Constraining page cache or moving to a genuinely
larger-than-RAM model is required to test storage savings.

Every policy produced the same logical digest. The source c4 campaign passed
12/12 compiled Rust tasks, but this experiment replays expert bytes rather than
matrix arithmetic. One physical run per policy means the elapsed break-even is
descriptive, not a confidence interval.

Data: [`gemma4-demand-cache-break-even.json`](data/gemma4-demand-cache-break-even.json).

## Decision

Use demand admission, start near 128 MiB for this workload, and grow only when
online reuse-distance telemetry justifies it. Do not preload from frequency
alone. Next validation needs either controlled RAM pressure or a model larger
than combined RAM/page-cache capacity, plus complete inference consuming the
application-owned buffer.

