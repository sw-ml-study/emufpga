# What this saves, and why MoE is the best case

The goal is not speed. It is **doing more with less at equal
correctness**: less VRAM, less system RAM, cheaper and older hardware,
fewer kWh. Several documents here hedge in a way that reads as doubt
about the thesis. The measurements do not support that doubt. They
support the thesis and are quiet about speed, which was never the
claim.

## The plain version

Conventionally every weight must be resident **before** any of them
can be used, because the engine will ask for them in an order that
depends on the input. That is the VRAM bill, and it is why a 300 GB
model needs 300 GB of fast memory.

Streaming says: put the weights on a conveyor belt. Each one passes
the compute once, in order, and the work is done as it goes by.
**Nothing is held.** The cost is that you cannot go back, so the
weights must be laid out in the order they will be wanted -- which is
what `layouts/*.order` and the no-seek tests exist to guarantee.

The payoff, and the reason concurrency matters: **if five clients are
waiting, one pass of the belt serves all five.**

## What has been demonstrated

Measured, not projected:

| | |
| --- | ---: |
| SmolLM2-135M weights on disk | 269 MB |
| the same, resident in RAM | **4 KiB** |
| ratio | **65,674x** |

- The streamed path is **bit-exact** against a conventional resident
  path on the real checkpoint (saga 2 step 6).
- Streaming costs about **2.5%** of wall clock at batch 32, against
  identical arithmetic.
- **Five clients asking five different questions**, prompts of
  different lengths, decoded from **one sweep per step**, every token
  matching `transformers` decoding that prompt alone (saga 2 step 11).
- `bf16` halved the traffic for **zero** accuracy cost, because the
  checkpoint was already bf16 and f32 storage was pure waste (saga 2
  step 9).
- Four model architectures -- TRM, HRM, BDH, SmolLM -- all verified
  against their references while streamed.

That list is the goal, achieved at small scale. What has *not* been
shown is that any of it is fast, and that was never the claim.

## Why MoE is the strongest case, not a stretch goal

A mixture-of-experts model has one defining problem, and it is a VRAM
problem specifically: **routing is data-dependent, so every expert
must be resident even though almost none of them will fire.** A 671B
MoE with 37B active still needs the whole 671B in memory, because you
cannot know in advance which experts a token will pick.

Streamed, that inverts. Each expert is its own stream. Only what
passes is read.

And the interesting part is what happens with concurrency. With `E`
experts and top-`k` routing, `N` concurrent requests want up to `N*k`
experts between them:

| concurrent requests | experts wanted, `E=256`, `k=8` |
| ---: | --- |
| 1 | 8 -- random access, which means disk **seeks** |
| 4 | up to 32 |
| 16 | up to 128 |
| **32** | **256 -- a full sweep is now FEWER bytes than fetching per request** |
| 64 | 256, and every extra client is free |

The crossover is `N = E / k`, which is **32** at these shapes. Past
it, sweeping every expert once costs less traffic than fetching the
ones each request asked for -- and a sweep is exactly what a cheap
sequential device is *good* at, where seeking is what it is worst at.

**Concurrency makes this better.** That is backwards from conventional
serving, where more concurrent requests mean more memory pressure, and
it is the whole argument. The batch converts data-dependent random
access into a predictable sequential scan.

Below the crossover the sweep is still often the right choice on
spinning media, because `N*k` scattered reads with seek latency can
lose to one contiguous pass long before they lose on bytes.

## One device per expert

Experts are independent: no expert needs another expert's weights.
So the work partitions with no shared state at all.

```
 activations in
      |
      +--> device 0: streams experts   0..31  --> partial sums
      +--> device 1: streams experts  32..63  --> partial sums
      +--> ...                                     |
                                                   v
                                            combine, route out
```

Each device sweeps its own stream at its own pace. Nothing is shared
but activations, which are kilobytes. This is the same partition saga
2 step 11 measured for a dense model -- weights amortize across
clients and do not scale with them, activations scale with clients and
carry no weights -- applied one level up.

## The honest limit: the KV cache

The one thing that cannot be streamed is per-conversation attention
state. It is written as generation proceeds, read on every subsequent
token, and it grows with both clients and context. It is the real
memory bill of this design.

For a 300 GB MoE with compressed attention state, roughly:

| clients | context | KV cache | weights resident |
| ---: | ---: | ---: | ---: |
| 1 | 4,096 | 0.29 GB | ~0 |
| 5 | 4,096 | **1.44 GB** | ~0 |
| 20 | 4,096 | 5.76 GB | ~0 |
| 5 | 32,768 | 11.51 GB | ~0 |

**300 GB of model, about 1.4 GB of RAM for five clients.** That runs
on an old 8 GB card, or on system RAM with no card at all. The KV
cache is roughly 200x smaller than the weights it replaces as the
binding constraint -- which is the entire point.

This is also the number that decides the design. It is why BDH
(saga 2 step 7) is a poor fit despite streaming perfectly: its working
state exceeds its whole weight set at long sequences. Check the
resident working set of any new architecture **before** planning a
rung for it.

## What is not yet true

Stated plainly, so the case above is not read as more than it is:

- **No MoE model has been run here.** The expert-sweep analysis is
  arithmetic over published routing shapes, not a measurement.
- **Ternary has never run against a real model.** `spm-codec-any`
  refuses the profile rather than supporting it. It is the change that
  would move both traffic and arithmetic at once
  (docs/research-ternary-fpga.md).
- **No disk has been measured.** Every store in every result has been
  page-cached RAM. The demanded-bandwidth figures are requirements a
  store would have to meet, not observations of one meeting them.
- **No energy has been measured**, though correct-answers per kWh is a
  stated goal.
- **The largest model run here is 135M parameters**, where nobody
  needs the savings. The savings are real and the scale is not yet.

## The order to attack these

Each is cheap and each can falsify the next:

1. **Measure a cold store.** Put a `.spm` on a real disk, drop the
   page cache, and re-run the serving demo. This is the cheapest
   experiment in the list and it tests the central assumption.
2. **Run ternary end to end** on a model trained ternary, against its
   reference. Predicted 5.31 MB per generated token at five clients,
   against 42.5 MB at bf16.
3. **Run one MoE layer**, streamed, with routing, and check the
   expert-sweep crossover against the arithmetic above.
4. **Then** scale up, having established that each mechanism holds.
