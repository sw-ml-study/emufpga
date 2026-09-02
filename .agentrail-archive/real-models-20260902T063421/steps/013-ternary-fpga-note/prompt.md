Answer, as a research note, whether a ternary ISA on an FPGA would
help the serial approach -- and whether an FPGA aux processor beside a
PC holding weights on disk is a sound partition.

Ground every claim in what this project has MEASURED. Mark projections
as projections. Do not quote an unsourced device resource count as
fact (CLAUDE.md).

The question decomposes into four:

1. Does ternary help the SERIAL approach specifically, or only the
   storage side? The measurements say we are compute-bound, not
   storage-bound, so a change that only shrinks bytes would not help.
2. Is "a LUT implements a trit" the right framing?
3. Where must the PC/FPGA boundary fall, and what crosses it?
4. What role would an MLPL-like description actually play?

Write docs/research-ternary-fpga.md. Say plainly what would kill the
idea as well as what supports it -- a research note that only supports
its own thesis is an advertisement.
