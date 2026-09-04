# Granite MoE routing distributions

This experiment replaces a uniform-independent routing assumption with routes
observed during the Rust full forward pass for the pinned Granite 3.1
1B-A400M Q6_K model. Reproduce it with `scripts/measure-granite-routing`.

## Scope

- Five hand-selected prompt domains: greeting, factual completion, Rust code,
  science explanation, and arithmetic.
- 23 prompt tokens across all 24 MoE layers: 552 token/layer routes and 4,416
  expert assignments.
- Top-8 of 32 experts. No generated tokens and no independent concurrent
  client sequences.
- Model SHA-256:
  `4566cfa92be10888026bd3663c83d64e91cd91f874dfb3607596587ff1c8f67f`.

The compact generated result is
[`docs/data/granite-routing-analysis.json`](data/granite-routing-analysis.json).
Raw per-route JSON remains under `/disk1/tmp/emufpga-granite-routing` because it
is reproducible intermediate data.

## Result

| Tokens grouped | Observed mean union | Layer/prompt samples | Independent formula | Observed traffic |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.00 | 120 | 8.00 | 25.0% |
| 2 | 13.51 | 96 | 14.00 | 42.2% |
| 3 | 16.79 | 96 | 18.50 | 52.5% |
| 4 | 18.31 | 96 | 21.88 | 57.2% |
| 5 | 20.70 | 96 | 24.41 | 64.7% |
| 6 | 21.00 | 48 | 26.30 | 65.6% |

For independent, uniform top-8 selection, the expected union is
`32 × (1 − (1 − 8/32)^B)`. Observed unions are smaller from B2 onward. At B6,
the difference is 5.30 experts: the observed selected-union schedule fetches
65.6% of expert bytes rather than the formula's 82.2%.

This is evidence of repeated/correlated routes in these prompt prefixes, not a
population estimate. Tokens in one causal prompt share context and are not the
same workload as B independent serving clients. The declining eligible corpus
also leaves only 48 layer samples at B6. We therefore display the observation
only for an exactly matching 32-expert/top-8/B1–B6 scenario and retain the
analytical projection elsewhere.

Expert selection was mildly skewed: the most frequent expert, 13, received
191/4,416 assignments (4.33%); expert 14 received 95 (2.15%). Uniform share
would be 3.125%. This small corpus is insufficient to justify a fixed expert
cache, but it makes cache hit-rate measurement a concrete next experiment.

## Repeated warm timing

[`docs/data/granite-timing.json`](data/granite-timing.json) contains seven-run
distributions for B1, B4, and B8. These are release scalar Rust measurements of
the block-0 expert sweep with a warm Linux page cache—not end-to-end inference.

| Batch | All-expert median tok/s | Selected-union median tok/s | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 10.98 | 12.23 | +11.4% |
| 4 | 15.75 | 15.97 | +1.4% |
| 8 | 17.59 | 17.62 | +0.1% |

At B1, selected-union reduces median warm read time from 14.17 ms to 3.80 ms,
but compute still takes roughly 69 ms. By B8 the measured throughput difference
nearly disappears. The result defends a memory-traffic benefit and rejects an
automatic throughput claim for the current scalar engine. It says nothing yet
about cold storage, GPU kernels, FPGA frequency, I/O overlap, power, or total
generation latency.
