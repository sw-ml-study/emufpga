Build the TRM importer: a real checkpoint in, a `.spm` file out.

Target: `yagizdevre/trm-maze-30x30`, 15 tensors, 6,824,450 parameters,
f32, 27.3 MB. Already downloaded and inspected -- see the saga plan.

SPLIT THE JOB AT THE PICKLE BOUNDARY, and here is why. A `.pt` file is
a ZIP holding a Python pickle plus raw storage blobs. Reading it in
Rust means writing both a ZIP reader and a pickle VM, and the pickle
subset torch emits is stereotyped but still ~25 opcodes. That is a lot
of code whose reuse is doubtful: the DeepSeek R1 1.58-bit quant this
project builds toward is GGUF, not pickle, so a `.pt` VM would be
written once and retired.

So:

1. `scripts/extract-checkpoint` -- Python, standard library only, NO
   torch dependency. Reads a `.pt` and writes a directory containing
   one raw little-endian blob per tensor plus a `manifest.tsv` giving
   name, shape and dtype per line. A working prototype of the pickle
   parsing already exists in this session's history: a
   `pickle.Unpickler` subclass overriding `find_class` and
   `persistent_load`. Handle f32 and bf16 -- TRM's two init vectors
   are bf16 while every weight matrix is f32.

2. `emufpga import` -- Rust. Reads the manifest and blobs, writes
   `.spm` with one f32 stream per tensor plus a sidecar manifest
   mapping stream index to tensor name and shape.

WHY A SIDECAR RATHER THAN NAMES IN THE FORMAT. `.spm` has no name
field and is not getting one. Names are a host concern: the importer
and the scheduler need them, the FPGA never does -- it streams bytes
in the order the directory declares. Putting names in the container
would make every consumer carry metadata only one of them uses. The
sidecar travels beside the file; document that they belong together.

Shape mapping: a 2-D tensor `(out, in)` becomes a stream with
`rows = out`, `cols = in`. Say what you do with 1-D tensors (the init
vectors and the bias) rather than letting the code decide silently.
Pick a group size for f32 and justify it -- the scale is inert for
this encoding, so the only thing group size controls is buffer
granularity.

TESTS MUST BE HERMETIC. Build small synthetic checkpoints in the test
itself; do NOT depend on the 27.3 MB download. `cargo test` stays fast
and works with no network. The real checkpoint is a manual
verification step whose result you record in docs/, not a fixture.

Also: weights never enter git. The output goes outside the tree or
under an ignored path, and `just check` will fail on any file over
1 MiB, so do not leave one lying in the working directory.

Verify against the real file manually and record in docs/:
- all 15 tensors present, shapes matching the inspection
- total parameters exactly 6,824,450
- the `.spm` payload byte count equals the sum over streams of
  `4 * rows * cols` plus 4 bytes per group for the inert scales
- round-trip: stream the file back and compare against the blobs

Gates: 25 LOC/function, 4 functions/module, 4 modules/crate, 350
LOC/file. `just check` green, `sw-checklist` 0 failed and no new
warnings, no `#[allow]`.

Do NOT implement TRM's forward pass. That is step 3.
