# What sw-mlpl would need to make these demos better

Written from friction actually hit while writing `spm_stream.mlpl` and
`spm_order.mlpl`, not from a wish list. Each item names the exact
workaround it would remove.

The demos live here and stay here. These are requests against
`../sw-mlpl`, ordered by how much they would improve this material.

## 1. WITHDRAWN: `disp` is not broken, my diagnosis was

**This entry was wrong and is kept rather than deleted**, because the
mistake is more instructive than the request was.

I observed that `disp("A"); print("B")` emits only `B`, and concluded
`disp` was silently failing. It is not. **`disp` returns a string** --
`type_of(disp([1,2,3]))` is `string` -- and it renders the ASCII box
the REPL and the final-value echo then display. The working idiom is:

```
print(disp([1,2,3]))
```

which prints the box mid-program, exactly as wanted. The glossary said
so all along. Only `examples/hello-world.mlpl`'s comment misleads, by
describing `disp` as something that prints.

So the correct ask is a **one-line documentation fix**, not a
behaviour change -- and a behaviour change would have been actively
harmful: `disp` appears in roughly 230 downstream files with recorded
output baselines, most ending in a `disp`, so making it write to
stdout would double their output and break every one of them.

**The lesson, which is postmortem 2's and which I failed here:**
resolve a surprise by asking the source of truth what a thing *does*,
rather than reasoning from the symptom. One call to `type_of` would
have answered it, and I had written that lesson down two steps
earlier.

## 2. No length-1 broadcast

```
[2] * [1, 2, 3]     ->  error: mul: expected [1], got [3]
2   * [1, 2, 3]     ->  2 4 6
```

A scalar broadcasts; a length-1 array does not. Both NumPy and APL
treat these the same way, and every indexing primitive here returns a
length-1 array rather than a scalar, so the two rules collide
immediately.

This is the single change that would most improve the demos. It is why
pulling activation `i` out of the stream needs an explicit collapse:

```
a = reshape(gather_rows(x, [i]), [])
```

With length-1 broadcast that is just `gather_rows(x, [i])`, and the
sweep reads the way the architecture actually works -- one activation
meets one row of weights.

## 3. No indexing primitive

```
v[2]    ->  parse error: UnclosedDelimiter
```

`gather_rows` needs a matrix, so reading element `i` of a vector means
reshaping to `[n, 1]`, gathering, and reshaping back:

```
reshape(gather_rows(reshape(order, [4,1]), [i]), [])
```

Four nested calls for `order[i]`. `spm_order.mlpl` does this twice per
recursion and it is the least readable thing in either file -- in a
demo whose whole purpose is to make traversal legible.

Either `v[i]` syntax or an `at(v, i)` builtin would collapse it. With
item 2 as well, it becomes `at(order, i)`.

## 4. A `dataflow` renderer

`docs/research2.txt` section 10 asks for this and it is the one
genuinely new capability the SPM curriculum needs. The current `svg()`
modes -- scatter, line, bar, heatmap, decision boundary -- are all
quantitative. The strongest SPM explanations are structural:

- the memory-hierarchy contrast (weights in VRAM beside KV and
  activations, versus weights streaming past a small resident set)
- the pipeline occupancy story: `storage -> FIFO -> lanes ->
  accumulators`, animated through store-bound, balanced and
  compute-bound
- the traversal contrast itself: a monotonic path through a heatmap
  with no reverse arrows

None of those are a line chart. Boxes, directed edges, groups, edge
labels, optional widths and highlight would cover all of them, and
would serve transformer diagrams, autograd graphs and compiler passes
just as well -- general enough to belong in sw-mlpl rather than here.

## Status

Against the sw-mlpl build in `../sw-mlpl`:

| ask | state |
| --- | --- |
| 1. `disp` | **withdrawn** -- doc fix only, and it is being made |
| 2. length-1 broadcast | applied upstream, not yet in the local build |
| 3. `at(v, i)` | planned |
| 4. `dataflow` renderer | planned |

Two things shipped that were not asked for here and are already in
use: **infix comparisons** (`i > 3` rather than `gt(i, 3)`, now in
both demos) and **qualified refs** (`:ns:name`).

When items 2 and 3 land, the activation lookup at the heart of
`spm_stream.mlpl` collapses from

```
a = reshape(gather_rows(x, [i]), [])
```

to `a = at(x, i)`, and the sweep reads the way the architecture works:
one activation meets one row of weights.

## What did NOT get in the way

Worth recording so nobody 'fixes' these:

- **Recursion as the loop.** The sweep is recursive, and that is the
  right shape: the point is that the reader only ever moves forward,
  and a recursive step makes that structural rather than a convention.
- **`each` and `reduce` with `:ref`.** `reduce(:add, v)` and
  `each(:relu, v)` both work once you know the function reference
  comes first.
- **`transpose`, `gather_rows`, `matmul`, `eq`, `relu`, `concat`.**
  Everything the arithmetic needed was already there. The gaps above
  are all ergonomic, not expressive -- the language can already say
  what the architecture does, which is the important part.
