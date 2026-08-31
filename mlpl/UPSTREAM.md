# What sw-mlpl would need to make these demos better

Written from friction actually hit while writing `spm_stream.mlpl` and
`spm_order.mlpl`, not from a wish list. Each item names the exact
workaround it would remove.

The demos live here and stay here. These are requests against
`../sw-mlpl`, ordered by how much they would improve this material.

## 1. `disp` prints nothing in script mode -- a bug

```
disp("A"); print("B")     ->  B
```

Only `print` produces output. `disp` appears to work in
`examples/hello-world.mlpl` **only because it is the last statement**,
where the script runner echoes the final value anyway. Its doc comment
says "disp prints any value; given a string it prints the text as-is",
and in a script that is not true of any statement but the last.

Either make `disp` write to stdout in script mode, or change
hello-world and the docs to use `print`. As it stands the first
example a newcomer reads teaches a function that does not do what it
says.

Cost here: every line of both demos uses `print`, so the friction was
ten minutes of confusion rather than a workaround. It is listed first
because it misleads readers, not because it cost the most.

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
