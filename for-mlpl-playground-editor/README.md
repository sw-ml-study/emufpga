# Run this in the browser

One self-contained file that demonstrates the whole Serial Parameter
Machine argument, with charts and diagrams. No downloads, no data
files, no local build.

## How

1. Open the playground:
   <https://sw-ml-study.github.io/sw-mlpl/>
2. Paste the contents of
   [`serial-parameter-machine.mlpl`](serial-parameter-machine.mlpl)
   into the editor.
3. Run.

Everything renders inline: two charts and three structural diagrams,
plus printed output at each step.

## What to look for

The file is six sections, and each ends in either **a difference that
must be zero** or a picture.

| section | the claim | how you can tell it is true |
| --- | --- | --- |
| 1 | a forward-only sweep computes exactly `W x` | `difference -- MUST BE ZERO` |
| 2 | ternary weights need no multiplier | same answer, add and subtract only |
| 3 | one sweep serves every waiting client | matches the conventional batched matmul |
| 4 | weights need not be resident | 65,811x fewer resident bytes |
| 5 | when the store is the bottleneck | a model whose residuals are ~0.1 s |
| 6 | recursion is a back-edge | re-read counts recovered from traffic |

Sections 1 to 3 compute their answers live, so the zeros are real
rather than quoted. Sections 4 to 6 plot **measurements** from a real
model -- SmolLM2-135M at bf16 -- recorded in
[`docs/results.md`](../docs/results.md).

The single most important line is in section 5:

```
MB/s demanded at 1 client, then at 5
1328 401
```

Bytes per sweep are fixed while compute grows with clients, so the
bandwidth a store must deliver **falls** as clients rise. The same
500 MB/s device is store-bound serving one client and entirely free
serving five. Serving more agents makes cheaper storage adequate,
which is the opposite of how conventional serving scales.

## Running it locally instead

```sh
../sw-mlpl/target/release/mlpl-repl -f \
  for-mlpl-playground-editor/serial-parameter-machine.mlpl
```

**Only the last picture renders this way.** The script runner shows
one value -- the final one -- so the earlier `svg` and `dataflow`
calls compute and are discarded. The browser playground and the
interactive REPL show every statement, which is why this file is
written for them. `mlpl/` holds the same material split into five
smaller lessons, one idea each, with the same caveat.
