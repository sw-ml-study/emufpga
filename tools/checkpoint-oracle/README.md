# Checkpoint extraction oracle

`extract.py` is the former production checkpoint extractor. It uses
only the Python standard library and remains here as an implementation
independent of the Rust path.

The normal entry point is `scripts/extract-checkpoint`, which is now
Rust-native. Run both implementations into separate directories and
use `diff -qr` when changing checkpoint parsing, numeric conversion,
or physical stream layout. The real TRM, HRM, and SmolLM checkpoints
were byte-identical across both implementations when the Rust path was
introduced.

The Rust `.pt` reader does not execute pickle callables, import Python
globals, or invoke `__reduce__`. It recognizes the narrow tensor
state-dict shape as inert metadata and rejects other top-level forms.
It also checks integer arithmetic and file bounds before allocating or
reading tensor payloads. This avoids Python pickle's arbitrary-code
execution path; malformed inputs can still be rejected as parser or
resource errors, so checkpoint provenance remains important.
