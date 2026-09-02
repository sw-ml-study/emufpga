Take up the sw-mlpl changes: simplify the two existing lessons and
write the two that were blocked.

All four asks in mlpl/UPSTREAM.md have landed -- `at(v, i)`, length-1
broadcast, infix comparisons, and the `dataflow` renderer. Verify each
against the local build rather than trusting the changelog, then use
them.

- Simplify `spm_stream.mlpl` and `spm_order.mlpl`. The predicted
  payoff was that the activation lookup collapses to `at(x, i)`; check
  that it does and that every difference still prints zero.
- Write `spm_residency.mlpl`: `Rp`, and the memory hierarchy drawn
  both ways. This needed `dataflow`.
- Write `spm_pipeline.mlpl`: when the store is the bottleneck. Use the
  MEASURED sweep from saga 2 step 12, not invented numbers, and
  re-derive `max(compute, bytes/rate)` in the language so the model is
  checked rather than asserted.

Update UPSTREAM.md's status, and say what would be useful next now
that nothing is blocked -- a request list is only honest if it empties
when the requests are met.
