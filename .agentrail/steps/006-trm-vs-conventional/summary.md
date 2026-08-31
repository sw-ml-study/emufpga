First head-to-head: TRM's forward pass streamed vs conventional resident, on the real 6,824,450-parameter checkpoint.

Built spm-trm-resident, the conventional path -- every parameter in
RAM, reached by subscript. Kept as a separate crate rather than a
generic parameter on spm-trm, because abstracting over 'weight source'
requires a trait that admits random access, and the streamed path's
guarantee is that random access is not expressible in its types.

Measured three configurations rather than two. Resident,
streamed-from-memory, streamed-from-file: the middle one keeps the
streaming discipline and drops the IO, without which a
file-vs-resident number conflates the cost of streaming with the cost
of storage and cannot be attributed to either.

All three agree bit for bit on the real checkpoint at every batch
size, asserted hermetically in tests/agreement.rs on synthetic
weights. Treated as a precondition rather than a result.

Numbers (M1 Max, release, best of 5, ms/forward): batch 1 gives
102/116/143, batch 8 gives 590/610/642, batch 32 gives 2402/2438/2461.
Parameter bytes resident: 27,297,800 against 4,096, a ratio of 1.5e-4
that is O(1) in model size.

Three findings, each written up with what it does not support:
streaming overhead falls monotonically with batch (41% to 2.5%); the
demanded store bandwidth falls from 2.9 GB/s at batch 1 to 166 MB/s at
batch 32, where a single SAS drive would keep up; and the 6,664x
memory win is real but not yet load-bearing, since 27 MB fits anywhere
and the resident path only stops fitting an 8 GB card around 2 G f32
parameters, ~300x up the ladder.

Two honesty items I made a point of recording rather than burying: the
+1.5% at batch 32 is at the noise floor (repeats move ~1%) and means
'no longer measurable here', not a precise figure; and the resident
path is the same scalar loop, not an optimised GEMM, so this measures
the streaming mechanism with arithmetic held fixed and says nothing
about beating a real inference engine.

Gate clean: 181 sw-checklist checks, 0 failed, 1 standing warning.
Pushed to main as 623b246.