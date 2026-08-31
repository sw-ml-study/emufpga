# Real Models

Vision: work UP a ladder of model sizes, proving the Serial Parameter
Machine path at each rung where the numbers are still checkable, before
attempting anything huge. Same discipline on hardware: one FPGA before
a mesh, one RP2350 before many.

The ladder: 7M (TRM maze) -> 27M (HRM) -> 120M -> 1 GB -> MoE -> huge.

Goal, restated so it does not drift: a LOWER-VRAM approach using
OLDER, CHEAPER hardware and older-but-good coding models. Not running
the newest model faster. VRAM is the resource that quadrupled; old
GPUs have capacity going spare and weak compute; old servers have
cheap system RAM, PMEM at un-inflated prices, and SAS. The bet is that
sequential streaming lets us spend the cheap resources instead of the
expensive one.

1. **encoding-aware-format** -- DONE. `.spm` byte length is per
   encoding; f32 profile added; ternary fixture untouched.
2. **trm-importer** -- DONE. Real checkpoint in, 6,824,450 parameters,
   zero byte mismatches.
3. **consumption-order-layout** -- DONE. Fixed step 2's alphabetical
   layout, which was the reverse of execution order. A forward sweep
   provably never seeks backward.
4. **trm-forward** -- run the forward pass over streamed weights.
   Streamed linear for stream N of M in f32, resident activation
   operators, recursion driving `rewind()`. Bit-exact against a
   resident reference.
5. **trm-vs-conventional** -- the first comparison. Same weights, same
   inputs, streamed path against a conventional resident path, with
   wall clock and energy where measurable.
6. **hrm-27m** -- second rung, second architecture shape.

Later rungs and the hardware ladder get planned when the rung below
them is done, not before.

Parked, deliberately, until the ladder reaches them:
- Rack experiments moving activations rather than weights over 10G.
- PMEM as the parameter store. 512 GB of PMem 200 holds a 300 GB model
  persistently, and PMEM prices have NOT quintupled -- it is the
  strongest cheap substrate available, in App Direct mode where
  sequential access plays to its strength.
- DeepSeek R1 1.58-bit. Baseline to beat: single-digit tokens/s on a
  Gen9 with 512 GB and an RTX 3060.
