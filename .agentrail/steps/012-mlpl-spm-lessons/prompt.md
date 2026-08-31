Write the SPM's central claim as an executable MLPL program, per
docs/research2.txt.

research2.txt argues sw-mlpl should become the executable teaching
language for this architecture while emufpga stays the research
vehicle, and names the most important demo: **same math, radically
different traversal**.

Write it, run it, and make every lesson end in a difference that must
be zero. A teaching artifact that cannot be executed is a diagram.

COVER

- `y = W x` conventionally, with the whole matrix addressable.
- `S = transpose(W)`: the mathematical view becoming the physical
  consumption view. That transpose IS the architectural change, and an
  array language is the right place to say so.
- A forward-only sweep that reproduces `y` exactly.
- Ternary as an instruction stream: `+1` add, `-1` subtract, `0`
  bypass, no multiply anywhere, still exact.
- Batch reuse: one sweep serving a whole batch, matching the
  conventional batched matmul. The stream is read once; only the
  arithmetic grows.
- The real TRM layout defect: alphabetical order forces backward
  seeks. Count them.

DISCIPLINE

Every `def u:` opens with a docstring, per sw-mlpl's CLAUDE.md. ASCII
only. Run both files and check the output before claiming anything.
