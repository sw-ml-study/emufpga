# Model install and expert profile

The install phase turns a supported MoE GGUF into a checksummed,
machine-readable inventory and a conservative cache proposal. It does not
rewrite weights, execute model content, or claim that a ranking improves
inference. JSON and bounded GGUF metadata replace Python pickle entirely.

## What is bound

`spm-model-profile install` records the complete model SHA-256, GGUF version,
architecture, exact expert tensor byte ranges, runtime build identity, hardware
profile and RAM/VRAM budgets. The outer envelope hashes the canonical JSON
payload. `verify` rejects a changed runtime or hardware profile before paying
the cost of hashing a large model, then rejects changed model bytes.

This protects against accidental stale reuse and detects modification; it is
not a signature and does not establish who supplied a model. GGUF parsing is
bounded and data-only. Rust avoids pickle's automatic object construction, but
Rust alone is not a security boundary: malformed-input bounds, checksums and
non-executable formats are the relevant controls.

```sh
cargo run --manifest-path components/cli/Cargo.toml \
  -p spm-model-profile -- install \
  --model MODEL.gguf --output profile.json --runtime LLAMA_BUILD_ID \
  --hardware docs/data/large12-hardware-profile.json \
  --ram-mib 4096 --vram-mib 12288 --trace calibration.log \
  --trace heldout.log

cargo run --manifest-path components/cli/Cargo.toml \
  -p spm-model-profile -- verify profile.json MODEL.gguf \
  LLAMA_BUILD_ID docs/data/large12-hardware-profile.json
```

Running `refresh` with fresh telemetry (the same validated path as `install`)
atomically replaces the output after the model and traces have been read. Each trace
digest is retained, so the input corpus is auditable. Runtime, topology, model,
or budget changes require a new profile. Unknown routes are demand-loaded; a
cold install has an empty preload list and never invents generic popularity.

## What the trace means

The current llama.cpp instrumentation emits selected expert counts for every
routed expert tensor operation. The profiler reports:

- assignments and unique layer/expert applications, plus the implied
  within-event reuse fraction;
- expert frequency and presence across trace files;
- the 64 most common same-layer co-activations;
- top-32 overlap between calibration traces and the final held-out trace.

The last trace is held out only for a stability diagnostic; all traces inform
the emitted cache proposal. A future cache-policy benchmark must freeze
calibration first and evaluate held-out work without learning from it.

## Current artifacts and limits

[`granite-install-profile.json`](data/granite-install-profile.json) proves the
cold path on Granite 3.1 1B-A400M Q6_K: 24 routed layers, 32 experts per layer
and eight selected per token. It emits no invented ranking.

[`gemma4-install-profile.json`](data/gemma4-install-profile.json) inventories
Gemma-4 26B-A4B Q5_K_M and uses independent Rust calibration and held-out
traces. Its preload/warm lists are proposals bounded by declared host RAM, not
measured cache wins. The SHA pass is bounded in memory, but reading 19.32 GB
from this HDD-backed path makes install slow.

On `large12`, the optimized install read and hashed the 19.32 GB Gemma file in
76.3 seconds; the 1.10 GB Granite control took 4.4 seconds. Gemma's two traces
contain 8,640 routed events and 143,280 assignments. Reusing one expert across
requests avoided 32.5% of otherwise repeated applications within those events,
and calibration versus held-out top-32 overlap was 90.6%. A 4 GiB host cache
budget proposed 477 static and 477 warm layer/expert resources. These numbers
measure routing structure and install cost, not policy throughput or disk bytes.

Correctness is unaffected today because the manifest is advisory and changes
no arithmetic. Existing evidence shows bit-identical logits for resident versus
reclaimed expert pages and passing compiled Rust tasks. The decisive next test
is for the scheduler to consume this placement, freeze it, and compare static,
warm, adaptive and cold-demand policies on unseen agent work. Success means
fewer physical bytes or less expert wait at identical outputs; a popularity
list that merely duplicates Linux page cache is a negative result.
