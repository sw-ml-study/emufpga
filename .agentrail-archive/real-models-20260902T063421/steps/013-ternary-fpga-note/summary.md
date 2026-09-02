docs/research-ternary-fpga.md: answered whether a ternary ISA on an
FPGA would help the serial approach, from measurements rather than
enthusiasm.

Yes, but not for the usual reason. Ternary is argued on storage
grounds, and that alone would not help because this project measured
twice that it is compute-bound. It helps because it moves both terms:
arithmetic intensity goes batch*0.25 -> batch*4.00 MACs per
weight-byte, AND the operation collapses from multiply to
select-and-add. The fabric model's 16 B/cycle, which fed 8 lanes at
f32, feeds 64 at ternary.

Sharpened the framing: storing a trit is two bits, but the win is the
operation, and the existing encoding's bits are control lines rather
than operands -- the stream is a two-bit instruction stream whose
program is the model. Recorded that 0b10 is a spare code point nobody
has examined.

On the PC/FPGA partition the numbers are decisive: weights must never
cross the host link, activations may. And the boundary falls exactly
where step 11's measurement already split a decode step, which is not
a coincidence -- attention has no weights for the same reason it does
not amortize.

Said plainly what would kill it: no ternary model has run here at all,
post-training ternary would likely destroy a 135M model so the
experiment needs one TRAINED ternary, BDH shows activations can exceed
the weight set anyway, and the partition assumes an independent
parameter path the boards may not have.