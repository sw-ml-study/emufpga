# Controlled storage-cache break-even

When model pages are prevented from remaining in Linux page cache, a small
demand-admitted expert cache produces a real but modest storage saving.

The source is Gemma-4 26B-A4B Q5_K_M on the single HDD in `large12`. After
every held-out c4 batch, `POSIX_FADV_DONTNEED` is applied to the model and zero
resident pages are verified. The 128 MiB userspace cache survives. This is an
equivalent controlled page-cache-bypass experiment, not `O_DIRECT`; failure to
evict aborts rather than silently falling back.

Across three recorded routing runs and six repeated batches, admit-on-second-
use saved 4.8% of physical reads (95% paired interval +/-2.2 percentage
points), 5.1% of expert-read wait (+/-2.5 points), and 4.5% elapsed time
(+/-2.1 points). It broke even on physical bytes by batch two in all runs.
Individual physical-byte savings were 3.8%, 5.1%, and 5.5%.

The price was roughly 128 MiB retained expert data and about 95-128 MiB higher
peak RSS. All paired expert-byte digests matched. This validates storage reuse,
not model arithmetic: expert computation and GPU work are absent, while the
source agent campaign supplies separate executable-task evidence.

The repeated identical batch favors stable routing, so 4.8% is closer to an
upper bound than a general workload forecast. Ordinary buffered runs saved no
physical bytes because this machine has enough RAM for Linux to do the caching.
The application cache matters only when page cache cannot retain the working
set, as with a model larger than RAM or deliberate memory pressure.

Data: [`gemma4-direct-cache-break-even.json`](data/gemma4-direct-cache-break-even.json).

## Decision

The idea bears limited fruit at the storage layer, but not enough yet to claim
an end-to-end serving advantage. Keep demand admission and the 128 MiB bound.
Next, make complete inference consume the same cache interface, then compare
correct results per hour and per kWh. SSD/SAS tiering becomes worthwhile only
if that integration preserves the measured physical-byte reduction.

