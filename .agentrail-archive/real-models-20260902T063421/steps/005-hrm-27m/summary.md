Rung 2 of the model ladder: HRM, 27M, from sapientinc/HRM.

Streamed HRM's two-module recursion and verified it against the
official implementation. A single contiguous [low][high] rotating
region serves both modules with plain rewind-to-zero: the last low
sweep of an outer cycle leaves the cursor exactly where the high
module begins, so no seek is needed and the WeightStream surface
stays seek-free.

Verification found a real defect. The port computed low(z_L) where
HRM computes low(z_L, z_H + input) -- the input embedding never
reached the recursion at all. Caught only by running the official
sapientinc/HRM ReasoningModule side by side, which needed name
remapping and a shim for flash_attn (CUDA-only). After the fix,
cosine 1.000000000000 against the official module, relative error
4.16e-7. Reference tooling preserved under tools/hrm-reference/.

Also hardened the postmortem's findings into the gate. Audited all
nine recorded defects for whether anything automated would fail if
the bug returned; six were covered, three were not -- and those three
(rms_norm axis, imported-matrix transpose, HRM injection) were
exactly the ones findable only by a manual torch reference run. Added
hermetic guards for all three, each verified by reintroducing its
defect and confirming a red test. scripts/test-extract runs
unconditionally in scripts/check because extract-checkpoint belongs
to no cargo workspace and nothing would otherwise gate it. Defect 3
(accumulator layout) is deliberately left unguarded and documented as
such: it is not observable through the public API and a timing
assertion under the shared build lock would be flaky both ways.

Running the documented 'just check-all' entry point surfaced a cargo
output filename collision that had been scrolling past: spm-trm and
spm-hrm both had an example named xcheck, and with one shared
target/ whichever built last won -- a TRM cross-check could have run
HRM's binary. Renamed trm-xcheck / hrm-xcheck / trm-bisect and put
the rule in CLAUDE.md beside the shared-target note that causes it.

156 tests. sw-checklist 176 passed, 0 failed, 1 warning (the standing
uninstalled-binary note). All six components gate clean.

Open: the flash_attn shim is the last assumption in the HRM
verification -- re-run tools/hrm-reference/official.py against real
flash-attn on a Linux/NVIDIA box to close it.