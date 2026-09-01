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
| 1. `disp` | **withdrawn** -- my misdiagnosis; doc fix only |
| 2. length-1 broadcast | **shipped** |
| 3. `at(v, i)` | **shipped** -- and it indexes matrix rows too |
| 4. `dataflow` renderer | **shipped**, groups/widths/highlight included |

Everything asked for has landed, and nothing here is blocked.

The two follow-ups below, raised after the renderer arrived, have also
shipped:

| follow-up | state |
| --- | --- |
| honest width channel for extreme ratios | **shipped** as `width_scale: "log"` |
| back-edge that reads as a rewind | **shipped**, dashed and routed below |

`spm_residency.mlpl` now passes real byte counts -- 269,564,308 against
4,096 -- rather than hand-picked stroke widths, which is only
renderable because of the log scale. `spm_rotation.mlpl` exists
because of the back-edge: a rewind is an edge pointing at an earlier
stage, and it now draws as one.

Two things shipped that were not asked for here and are already in
use: **infix comparisons** (`i > 3` rather than `gt(i, 3)`, now in
both demos) and **qualified refs** (`:ns:name`).

The predicted payoff arrived exactly as described. The heart of
`spm_stream.mlpl` went from

```
row = reshape(gather_rows(S, [i]), [3]);
a   = reshape(gather_rows(x, [i]), []);
```

to

```
row = at(S, i);
a   = at(x, i);
```

which is one activation meeting one row of weights, and is now
readable as the thing it describes. `dataflow` unblocked the two
lessons -- residency and pipeline occupancy -- that no line, bar or
heatmap could draw.

## What would be useful next

Two, both found by a diagram coming out scrambled in the playground.
The first is a defect; the second is a missing knob.

### 5. Labels are placed without checking what is already there

Two distinct collisions, same root cause -- the renderer places a
label and never asks whether that rectangle is occupied.

**An edge label is centred in `COL_GAP` (96) without applying
`CHAR_W`.** The renderer already measures text to size node boxes; it
just does not use that measurement here, so any label over 12
characters lands on the boxes either side. Two labels here were 160px
and 152px in a 96px gap.

**An edge that skips a layer puts its label on the node it passes.**
`RAM -> compute` crossing layer 1 drew "resident" underneath the
buffer box. This is the harder one: shortening the label does not help,
because the collision is with a node the edge routes past.

Either would be fixed by testing the label rectangle against the boxes
already placed and nudging, or by routing skip-edges around rather
than through. `mlpl/verify-diagrams` in this repo is the stopgap: it
renders every `dataflow` call in a file separately and fails on any
overlap. That a caller has to re-parse the renderer's SVG to discover
whether it collided is the argument for fixing it upstream.

### 6. No control over canvas size or spacing

`NODE_W`, `CHAR_W`, `BOX_PAD`, `NODE_H`, `COL_GAP`, `ROW_GAP` and
`PAD` are fixed constants and the canvas derives from content. There
is no way to ask for a wider diagram or more room between boxes, which
is the first thing anyone reaches for when a picture is crowded. An
optional multiplier on the gaps, or explicit `col_gap` / `row_gap`,
would cover it without changing any existing call.

### 7. The arrowhead scales with stroke width

A wide edge renders as a large triangle rather than a thick line, and
at `W_MAX` (9) it occludes what is behind it. A marker that stays a
fixed size while the line thickens would read better.

### Correcting request 4: width is the wrong channel for a big ratio

I asked for the log-width scale and it works as specified. Then using
it taught me the request was aimed at the wrong thing.

**A stroke width cannot express 65,000:1 however it is scaled.**
Nobody reads a thickness and recovers a ratio, and the log scale that
makes the extremes renderable also compresses them into looking
similar. The honest channel is the label: these diagrams now say
"269 MB" and "4 KiB" on the edges and keep widths modest, and they
communicate the ratio far better than any stroke did.

The feature is still right for modest ratios, where a reader really
can compare two thicknesses. It is recorded here because the request
was granted in good faith and the premise behind it was mine and
wrong.

### Also worth documenting: two layout rules callers cannot discover

Neither is a defect, but both cost a round of unreadable output:

- **A group band spans every layer its members occupy**, so grouping
  nodes at different depths overlaps another group's band.
- **An edge that skips a layer routes through it**, so its label
  collides with whatever is there.

Both are consequences of the layering, and both are invisible until
the picture comes out wrong.

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
