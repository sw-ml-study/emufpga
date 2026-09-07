# Profile-driven cache results

## Verdict

The frozen 512 MiB cache profile is a negative result for short, cold agent
windows. It finds experts that recur, but loading them before demand costs more
I/O than the subsequent hits save. This does not reverse the measured benefit
from sharing one selected-expert traversal among concurrent agents; it rejects
speculative preload as the next optimization on this workload.

The physical replay used the exact Gemma-4 26B-A4B Q5_K_M byte ranges and
request-time routing from the previously passing Rust agent campaign. The
profile was trained only on the separate c4 calibration trace. Each policy ran
three times after per-file eviction, for the first 32 routed layer events.
Preload time and bytes are included.

| agents | policy | median hit rate | physical bytes vs demand | median time |
| ---: | --- | ---: | ---: | ---: |
| 1 | static | 15.9% | +9.8% | 17.08 s |
| 1 | warm | 27.6% | +22.0% | 18.71 s |
| 2 | static | 14.4% | +8.6% | 18.93 s |
| 2 | warm | 26.5% | +18.6% | 20.36 s |
| 4 | static | 12.9% | +9.5% | 18.52 s |
| 4 | warm | 24.0% | +21.6% | 21.98 s |
| 8 | static | 12.6% | +8.4% | 23.46 s |
| 8 | warm | 23.4% | +18.4% | 24.72 s |

Demand and adaptive results were effectively identical. Adaptive admits on the
second observation, but these short windows did not reach a third observation,
so it avoided preload cost without earning hits. All 48 policy/run digests
matched their demand controls. This is byte-path correctness, not new model
quality evidence; the source traces came from the earlier executable campaign.

The raw trials remain under `/disk1/tmp/emufpga-profile-cache-campaign-v2` and
the checked result is
[`gemma4-profile-cache-analysis.json`](data/gemma4-profile-cache-analysis.json).
Device reads closely tracked process reads, making unrelated-I/O contamination
unlikely, though device counters are still whole-device observations.

## Next decision

Do not preload a broad popularity tier for short sessions. The easier and more
promising next policy is demand-only admission coordinated across requests,
with a cache surviving many agent batches. Measure its break-even lifetime and
phase drift before adding SSD promotion. A longer-lived service may amortize
the same static load; if it does not, frequency alone is the wrong signal and
reuse distance/recency must dominate. Multi-SAS remains a later hardware test.

