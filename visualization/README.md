# Serial-MoE FPGA visual emulator

This is a visual emulator of the purpose-built accelerator proposed by
`emufpga`, not a general FPGA editor or floorplan viewer.

Generate an authoritative trace:

```sh
SPM_GRANITE_TRACE_JSON=visualization/trace.json \
  scripts/verify-granite-moe-full
```

Then serve the repository root and open `/visualization/`:

```sh
scripts/serve-visualization
```

The helper uses the Rust `basic-http-server` binary and binds only to localhost.

The hosted copy is built locally and committed before deployment:

```sh
scripts/build-pages
git diff --exit-code -- pages
```

Publish the committed output with `scripts/deploy-pages`. As in the neighboring
`sw-mlpl` repository, this pushes only the generated subtree to `gh-pages`; no
GitHub runner compiles or transforms the application. Once Pages uses that
branch, the result is available at <https://sw-ml-study.github.io/emufpga/>.

The UI has a built-in measured block-0 fallback, so it remains demonstrable
without the model artifact. `trace.json` is generated and ignored by git.

The browser uses dependency-free SVG and JavaScript because the animation is a
small deterministic state machine. This reuses the neighboring `ml-viz` HUD,
SVG callout, and Rust-authored-scene pattern without importing its three.js
stack. MLPL remains a good companion for static charts, but its current SVG
surface does not provide the event loop needed for this animation.

The proposed-hardware explorer is intentionally broader than the FPGA board:
it distinguishes packed weights, prompts/activations, KV traffic, and control
signals across five experiment topologies. Moving particles indicate direction,
not rate, and every unmeasured topology is labeled as such.

See [the ELI5 serial-processing guide](../docs/serial-processing-eli5.md) for
weight, activation, KV-cache, partial-sum, training-state, and hybrid streaming
approaches represented by this datapath.
