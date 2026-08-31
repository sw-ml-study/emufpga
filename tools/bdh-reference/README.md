# BDH reference comparison

What produced the numbers in `docs/results.md`, "BDH, rung 3".

`reference.py` runs the official BDH from
[`pathwaycom/bdh`](https://github.com/pathwaycom/bdh) (arXiv
2509.26507) and dumps its weights in `.spm` stream order plus the
output of every stage, so `spm-bdh`'s streamed engine can be checked
against it one stage at a time.

`bdh.py` is NOT vendored here. Fetch it beside this file:

```sh
curl -sLO https://raw.githubusercontent.com/pathwaycom/bdh/main/bdh.py
```

## Running it

Weights never enter this repository, so work outside the tree.

```sh
uv venv && source .venv/bin/activate && uv pip install torch
python reference.py out 16
emufpga import -i out -o bdh.spm --order layouts/bdh.order
cargo run --release -p spm-bdh --example bdh-xcheck -- out bdh.spm
```

## Two things worth knowing before changing it

**`model.eval()` is load-bearing.** BDH's default dropout is 0.1, and
a training-mode forward pass is not reproducible. Every comparison
here would go noisy for a reason that has nothing to do with
streaming.

**Do not run `scripts/extract-checkpoint` over BDH.** That extractor
transposes every 2-D tensor, which is correct for a torch `Linear`
storing `(out, in)`. BDH stores every parameter as `(in, out)`, so its
raw row-major bytes already are `.spm` stream order and the declared
shape is simply reversed. Passing it through the generic path would
reintroduce postmortem defect 8 in mirror image -- and, as that defect
showed, a byte round-trip test would not notice.

## Seeded random weights, not a checkpoint

`pathwaycom/bdh` ships architecture and training code but no trained
model, so `reference.py` seeds the init and compares against that.
Every formula is verified; nothing about BDH's quality is. If a
checkpoint is published, rerun against it.
