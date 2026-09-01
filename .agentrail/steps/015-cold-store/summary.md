Measured what happens when the parameter store is not RAM -- the
cheapest experiment that could have falsified the thesis.

spm-stream-throttle delivers a stream at a chosen rate and reports
stall time, which is eta observed rather than modelled. Bandwidth
only, stated as such.

The prediction (knee at 258 MB/s) was WRONG and usefully so: 500 MB/s
is free and 258 already stalls 35%. The 258 came from smol-xcheck's
8-position prefill applied to a 1-token decode step, so the failed
prediction located an apples-to-oranges comparison sitting in
results.md.

Measured at five clients over 2,760 MB: 500 MB/s free, 258 MB/s 35%
stalled, 150 MB/s 62%, 50 MB/s 87%. Every client's tokens matched the
reference at every speed -- a slow store makes this slower, never
wrong, which is the property that matters given the goal.

wall = max(compute, bytes/rate) fits every row within 2%, so the knee
is 401 MB/s for this engine at five clients: SATA-class SSD free,
spinning disk works at 2.7x.

THE RESULT THAT MATTERS: concurrency lowers the class of device you
need. One client demands 1328 MB/s and stalls 60% on a 500 MB/s store;
five clients demand 401 and stall 0% on the same store. Bytes per
sweep are fixed while compute grows with clients. Serving more agents
makes cheaper storage adequate -- the opposite of conventional
serving, and step 11's amortization reappearing as a hardware
requirement.

Not tested: a genuinely cold read. The throttle paces a page-cached
file, so first-touch latency is still unmeasured and needs sudo purge.

Gate clean across all seven components: 227 checks. Pushed bb714a0.