Write docs/why-this-saves-ram.md: the plain-language case, and the MoE
analysis that has not been written down anywhere.

The goal of this project is stated wrong in several places. It is not
speed. It is **doing more with less at equal correctness** -- less
VRAM, less system RAM, cheaper hardware, fewer kWh. Several documents
hedge in a way that reads as pessimism about the thesis when the
measurements actually support it.

WRITE, IN PLAIN LANGUAGE FIRST

- What has to be in RAM conventionally, versus streamed. Use the
  measured 269 MB -> 4 KiB.
- What we have demonstrated, as distinct from what we have not. The
  demonstrated list is longer than the documents currently imply.

THEN THE MoE ANALYSIS, WHICH IS NEW

MoE is the strongest case for this architecture, not a stretch goal,
and nothing in docs/ says so yet:

- MoE's defining problem is that all experts must be resident because
  routing is data-dependent. That is a VRAM problem specifically.
- Streamed, each expert is its own stream and only what passes is
  read.
- With E experts and top-k routing, N concurrent requests want up to
  N*k experts. Past N = E/k, a full sweep of every expert is FEWER
  bytes than fetching per request -- and a sweep is what a disk is
  good at, where seeking is what it is bad at.
- So concurrency makes this BETTER, which is backwards from
  conventional serving and is the whole argument.
- Experts are independent, so device-per-expert is embarrassingly
  parallel with only activations crossing.

THE HONEST LIMIT

The KV cache is the one thing that cannot be streamed: per
conversation, growing with clients and context. Give the numbers for a
300 GB MoE, and note it is roughly 200x smaller than the weights,
which is the point.

DO NOT OVERSTATE

No MoE model has been run here. Ternary has never run. Every store has
been page-cached, so no disk has actually been measured. Say so.
