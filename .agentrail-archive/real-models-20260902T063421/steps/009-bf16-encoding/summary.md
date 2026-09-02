A bf16 encoding profile, end to end, halving traffic on a real model.

Four rungs were verified and every one streamed f32, though
SmolLM2-135M's checkpoint is bf16 and the extractor widened it -- a
self-inflicted 2x on the axis this project exists to improve.

THE REAL DEFECT FOUND: the encoding discriminant was stamped and
ignored in four places, and spm_linear::take called the f32 codec
without consulting the descriptor at all. A bf16 stream would have
been decoded as f32 and produced plausible garbage with no error
anywhere. The Encoding doc comment already warned about that exact
shape of bug and it was still true one layer down. GroupView now
carries its encoding and spm-codec-any is the single place that knows
every profile.

Measured: .spm 538,594,324 -> 269,564,308 bytes; per-forward traffic
424,673,280 -> 212,336,640; demanded bandwidth 507 -> 258 MB/s at 8
positions and 143 -> 70.6 MB/s at 32. Exactly half on every traffic
row. A 135M model at batch 32 now demands what a spinning disk
delivers.

The accuracy cost is ZERO, and I checked why rather than accepting it:
the checkpoint is natively bf16, so widening and rounding back is
lossless -- 0 of 576 values differ in a directly compared layer. The
f32 profile was storing bf16 data in twice the space. Waste removed,
not a size-versus-accuracy trade. My own step prompt predicted a cost
and said exact agreement would be suspicious, so I verified the file
carries discriminant 3 on disk and added a test asserting that values
needing more than 8 mantissa bits DO get rounded -- if the bf16 path
were secretly f32, it fails.

Rounding is round-to-nearest-even on both sides, Rust and Python, with
a test measuring that it beats truncation on a long sum.

Gate clean: 211 checks, 0 failed, 1 standing warning, all six
components. Pushed af7882c.