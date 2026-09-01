# The Serial Parameter Machine, in MLPL

Executable explanations of what `emufpga` does, written in
[sw-mlpl](../../sw-mlpl). The Rust here is the research vehicle; these
are the teaching artifacts, and they run.

```sh
../sw-mlpl/target/release/mlpl-repl -f mlpl/spm_stream.mlpl
../sw-mlpl/target/release/mlpl-repl -f mlpl/spm_order.mlpl
```

**Want to just look at it?** `../for-mlpl-playground-editor/` holds a
single self-contained file to paste into the browser playground, where
every chart and diagram renders. Under `-f` only the **last** picture
in a file is drawn, because the script runner shows one value; the
playground and the interactive REPL show every statement.

## Why MLPL for this

The central claim is a claim about **layout**: the arithmetic of
inference does not change, only the order the weights are touched in.
An array language makes that claim checkable in a few lines, because a
layout transformation is a first-class expression rather than
something buried in pointer arithmetic.

`transpose(W)` is the whole architectural change, and in MLPL you can
write it, sweep the result forward only, and subtract the two answers
to show they are the same.

## The files

| file | the one idea |
| --- | --- |
| `spm_stream.mlpl` | a forward-only sweep computes exactly `W x`; ternary needs no multiplier; one sweep serves a whole batch |
| `spm_order.mlpl` | consumption order is not a filing preference -- alphabetical layout forces backward seeks |
| `spm_residency.mlpl` | what actually has to be in RAM: `Rp`, and the memory hierarchy both ways |
| `spm_pipeline.mlpl` | when the store is the bottleneck, from the measured store sweep |
| `spm_rotation.mlpl` | a rotating parameter store is a stream with one back-edge -- and re-reading is traffic, not reuse |

The first two end by printing a difference that must be all zero --
**same arithmetic, different memory contract.** The last two plot
measurements from `docs/results.md` rather than invented numbers, and
`spm_pipeline.mlpl` re-derives the `max(compute, bytes/rate)` model in
the language and prints its residuals against the measured run.

## What each demonstrates

`spm_stream.mlpl`:

- `y = W x` the conventional way, with the whole matrix addressable.
- `S = transpose(W)` -- the mathematical view becomes the physical
  consumption view. Row `i` of `S` is every weight activation `i`
  touches.
- A recursive sweep that only ever moves to row `i + 1`, reproducing
  `y` exactly. No address is computed and no row is revisited.
- The same sweep with `+1 -> add`, `-1 -> subtract`, `0 -> bypass`
  and **no multiply anywhere**, still exact. This is why the target
  fabric is decode/add/sub/accumulate rather than a multiplier array.
- One sweep serving a batch of three, matching the conventional
  batched matmul. The stream is read once; only the arithmetic grows.
  That is scan productivity `Ps`, and it is the economic argument for
  the whole approach.

`spm_order.mlpl` replays a real defect. The first TRM importer used
Python's `sorted()`, so the file was alphabetical while the forward
pass wanted `qkv, o, gate_up, down`. Sweeping that file means three
backward seeks, which the parameter stream cannot express -- `rewind`
returns to zero and there is no offset to give it. The program counts
the violations: zero for execution order, three for alphabetical.

The fix was a layout file rather than a smarter reader; see
`layouts/trm-maze-30x30.order` and
`components/cli/crates/spm-order/tests/no_seek.rs`.

## Where these live

Here, and they stay here. `docs/research2.txt` floated moving them to
`sw-mlpl/demos/`; the decision is to keep them beside the defect and
the layout file they explain, so a change to either is a change to
both.

What sw-mlpl would need to make them better is recorded in
[UPSTREAM.md](UPSTREAM.md) -- written from friction actually hit, not
from a wish list, and including one ask that turned out to be my own
misdiagnosis rather than a defect. The short version: a length-1 array
does not broadcast where a scalar does, there is no indexing
primitive, and the structural diagrams this material wants need a
`dataflow` renderer that does not exist yet.

All six asks in UPSTREAM.md have now landed upstream -- `at(v, i)`,
length-1 broadcast, infix comparisons, and the `dataflow` renderer --
plus a log-width channel and dashed back-edges, so every lesson the
plan called for is written and the earlier ones are simpler for it. The sweep now reads `row = at(S, i); a = at(x, i)`,
which is one activation meeting one row of weights.
