# Real Models

Vision: run a real trained model through the Serial Parameter Machine
path and compare it against the way you would actually run it today.
Saga 1 proved the machine on synthetic ternary matrices; saga 2 puts a
real model through it.

Target: `yagizdevre/trm-maze-30x30` -- TRM, 6,824,450 parameters, two
layers, f32 weights, solving 30x30 hard mazes. Chosen because it is
the same task as the Rust TRM in `softwarewrighter/train-trm`, so the
two are directly commensurable, and because its recursion
(H_cycles 3, L_cycles 4, halt_max_steps 16) re-streams the entire
weight set up to 192 times per puzzle -- the rotating parameter store
from the research, arriving for free.

Deliberately NOT in scope: quantization. The checkpoint is f32 and
stays f32, so the comparison is numerically clean by construction and
no accuracy question arises. Ternary belongs to the DeepSeek R1
1.58-bit quant (~300 GB), later, on the Linux box, where residency
actually bites.

1. **encoding-aware-format** -- `.spm` currently hardcodes ternary's
   two-bits-per-weight everywhere group sizes are computed. Make byte
   length a function of the encoding, add an f32 profile, keep the
   ternary golden fixture byte-identical.
2. **trm-importer** -- read a PyTorch `.pt` (zip + pickle) with no
   torch dependency, write `.spm` plus a sidecar manifest naming the
   streams. `emufpga import`.
3. **trm-forward** -- the operator set TRM needs beyond GEMV: RMSNorm,
   attention, SwiGLU MLP, and the recursion loop driving `rewind()`.
   Bit-exact against a reference forward pass.
4. **gpu-baseline** -- run the published model on Metal, compare
   against the SPM path on the same weights. Real wall clock both
   sides.
5. **recursion-reuse** -- extend `Ps` to count reuse across recursion
   depth, not only across batch. TRM re-reads its weights up to 192
   times per puzzle; the current metric cannot see that.
6. **overlapped-fetch** -- deferred from saga 1's plan. `eta` still
   measures a serial pipeline. Do it once there is a real workload
   whose consumption rate makes the overlap measurable.

Later, not yet scheduled: a Sudoku TRM (needs training -- roughly 1-3
weeks on this M1 Max, or ~$15 of rented L40S time, and a partial
checkpoint is enough to validate streaming); then HRM at 27M
(`sapientinc/HRM`), which is where a streaming-vs-resident comparison
starts to be worth making.
