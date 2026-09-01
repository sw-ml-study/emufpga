# Continuing this work on a Linux/CUDA box

Everything is on GitHub. This is what to fetch, what only works there,
and what is queued.

```sh
git clone git@github.com:sw-ml-study/emufpga.git
cd emufpga && just check-all      # should be 227 passed, 0 failed, 1 warning
```

The single warning is `sw-install`: the `emufpga` binary is not
installed. That is expected and the checkpoint process forbids running
`sw-install` unasked.

## Weights are not in the repository

By design -- see CLAUDE.md, "Large files never enter git". Fetch them
outside the tree. Every result in `docs/results.md` used one of these
four.

| model | source | size |
| --- | --- | ---: |
| TRM 7M | `yagizdevre/trm-maze-30x30`, checkpoint `step_390620` | 27 MB |
| HRM 27M | `zbloss/HRM-sudoku-extreme` | 109 MB |
| BDH | none -- `pathwaycom/bdh` ships code, not weights | -- |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | 269 MB |

```sh
mkdir -p ~/spm-weights && cd ~/spm-weights

# SmolLM2-135M -- the model every recent result uses.
curl -sLO https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
curl -sLO https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/config.json

# TRM and HRM, if you want to re-run the earlier rungs.
huggingface-cli download yagizdevre/trm-maze-30x30 --local-dir trm
huggingface-cli download zbloss/HRM-sudoku-extreme --local-dir hrm

# BDH has no checkpoint: reference.py seeds its own random weights.
curl -sLO https://raw.githubusercontent.com/pathwaycom/bdh/main/bdh.py
```

### Or copy them across, which is faster and removes doubt

The checkpoints are already downloaded on the Mac. Copying guarantees
the target sees byte-identical inputs, so any difference in results is
the machine and not the download.

**They currently live in a session scratchpad under `/private/tmp`,
which is ephemeral.** Move them somewhere durable first:

```sh
SP=/private/tmp/claude-501/-Users-mike-github-sw-ml-study-emufpga/\
dbb9c529-bbc8-428f-ad59-18c8934bbcc0/scratchpad
mkdir -p ~/spm-weights
cp -R $SP/smol $SP/trm $SP/hrm $SP/bdh ~/spm-weights/
```

Then copy only the **sources** -- 387 MB, against 3.3 GB for the whole
scratchpad:

```sh
scp ~/spm-weights/smol/model.safetensors \
    ~/spm-weights/smol/config.json      box:~/spm-weights/smol/
scp ~/spm-weights/trm/model.pt          box:~/spm-weights/trm/
scp ~/spm-weights/hrm/model.safetensors \
    ~/spm-weights/hrm/config.json       box:~/spm-weights/hrm/
scp ~/spm-weights/bdh/bdh.py            box:~/spm-weights/bdh/
```

| what | size | copy it? |
| --- | ---: | --- |
| `smol/model.safetensors` | 257 MB | yes -- the model every recent result uses |
| `hrm/model.safetensors` | 104 MB | yes, if re-running rung 2 |
| `trm/model.pt` | 26 MB | yes, if re-running rung 1 |
| `bdh/bdh.py` | 8 KB | yes -- upstream may have moved on |
| `*/extracted*`, `*.spm` | 644 MB | **no** -- regenerate |
| `bdh/out` stage dumps | 101 MB | no -- `reference.py` reseeds them |

**Regenerate the derived files on the target rather than copying
them.** They are as large as the sources, so there is no bandwidth
saved, and regenerating exercises `extract-checkpoint` and `emufpga
import` on the new machine -- which is a real check that the toolchain
works there. If the regenerated `.spm` differs byte-for-byte from the
one here, that is a finding worth chasing, and copying would have
hidden it.

Each `tools/*/README.md` has the exact pipeline for that model.
Python side: `uv venv && source .venv/bin/activate && uv pip install
torch transformers` (never plain `pip`).

## What only works on CUDA

`docs/plan.md` section 8b has the detail. The short version: one thing.

**`flash-attn`, to run `sapientinc/HRM` unmodified.** Its
`models/layers.py` imports `flash_attn_func` at module scope with no
fallback, so the HRM comparison here used a stand-in built on torch's
`scaled_dot_product_attention`. That shim is the **last untested
assumption** in the HRM verification -- it computes the same function,
but nobody has confirmed it against the real kernel.

Closing it is one command and one re-run:

```sh
uv pip install flash-attn --no-build-isolation
python tools/hrm-reference/official.py ...      # without the shim
```

Everything else -- import, streaming, all four rungs, serving, the
store sweep -- is pure CPU and runs identically.

## Two experiments that are EASIER on Linux

**The cold read.** `docs/results.md` measures a bandwidth-limited
store but has never measured a genuinely cold one, because dropping
the page cache on macOS needs `sudo purge` and a password. On Linux it
is scriptable:

```sh
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
cargo run --release -p spm-batch --example serve-demo -- \
    out smollm-bf16.spm extracted-bf16 8
```

That measures first-touch latency, which the throttle cannot model.
It is the cheapest remaining experiment.

**Twenty clients.** The model says demand falls as 1/N, so twenty
clients should need about 100 MB/s where five need 401. A bigger box
makes the compute side less of a constraint.

## What is queued

In the order `docs/why-this-saves-ram.md` argues for, cheapest first:

1. **A genuinely cold store** -- above.
2. **Ternary end to end.** The profile has existed since saga 1 and
   has never run against a real model; `spm-codec-any` refuses it
   explicitly. It needs a scale-aware accumulator the streamed path
   does not have, and a model **trained** ternary rather than
   quantized after the fact. Predicted 5.31 MB per generated token at
   five clients against 42.5 at bf16.
3. **One streamed MoE layer**, with routing, to test the expert-sweep
   crossover: past `N = E/k` concurrent requests a full sweep costs
   fewer bytes than fetching per request. This is where the larger
   hardware actually matters.
4. **Larger models.** The largest run here is 135M, where nobody needs
   the savings. `docs/why-this-saves-ram.md` projects a 300 GB MoE at
   about 1.4 GB of RAM for five clients.

## The MLPL demos need a second repo

`mlpl/` and `for-mlpl-playground-editor/` run against
[sw-mlpl](https://github.com/sw-ml-study/sw-mlpl):

```sh
git clone git@github.com:sw-ml-study/sw-mlpl.git ../sw-mlpl
cd ../sw-mlpl && cargo build --release
```

Or skip it entirely and paste
`for-mlpl-playground-editor/serial-parameter-machine.mlpl` into
<https://sw-ml-study.github.io/sw-mlpl/>, which needs nothing local.

## Read these first

- `docs/why-this-saves-ram.md` -- the goal and what is demonstrated.
- `docs/results.md` -- every measurement, with what it does not
  support. Note the CORRECTED banner partway down: an early conclusion
  about recursion was wrong and is annotated in place.
- `docs/postmortem-1.md` and `-2.md` -- fourteen defects and what
  caught each. The standing rule: **build the reference comparison
  before the streaming path.** The rung that followed it verified on
  the first run with no bugs; the two that did not each shipped bugs.
