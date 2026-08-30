# Code metrics and complexity gates

Adapted from `../sw-mlpl`. This is the canonical architecture guide for
emufpga: it defines the metric gates, when to refactor, and where new
code belongs. `sw-checklist` enforces the fallback floor; the targets
below are stricter and are what we actually aim for.

## 1. The gates

| Unit | Target | sw-checklist WARN | sw-checklist FAIL |
| --- | --- | --- | --- |
| Function | 25 LOC | > 25 LOC | > 50 LOC |
| Module | 4 functions | > 4 functions | > 7 functions |
| Crate | 4 modules | > 4 modules | > 7 modules |
| File | 350 LOC | > 350 LOC | -- |
| Component | no automated gate | -- | -- |

The module and crate targets are 4, not 5: `sw-checklist` warns
*above* 4, so targeting 5 guarantees a warning. `lib.rs` counts as one
of a crate's four modules, which in practice means **a facade plus
three modules of behavior**. That is tight on purpose -- it is the
pressure that produces small single-purpose crates.

Only `src/` is measured. Integration tests under `tests/` are not
counted, so test files may hold as many helpers and cases as
readability wants. Inline `#[cfg(test)] mod tests` blocks in `src/` ARE
counted, which is a second reason to prefer `tests/`.

There is no automated gate on crates per component. `components/format`
runs to six crates because the four-module ceiling pushed the header,
codec, layout, cursor and file concerns apart. Let the module gate
decide the crate count rather than picking a crate count first.

Rust edition 2024. Every component workspace sets
`[workspace.lints.clippy] pedantic = "warn"`, and
`cargo clippy --all-targets --all-features -- -D warnings` must be
clean.

The 25 LOC budget assumes idiomatic formatting. Do NOT try to reach it
with whitespace tricks, `#[rustfmt::skip]`, or `#[allow(...)]`.
Suppressing the measurement is not the same as meeting it.

## 2. Why this repo spreads horizontally

There is no root `Cargo.toml`. Every directory under `components/` is
its own cargo workspace. This is structural, not cosmetic: the
5-crates-per-component and 5-modules-per-crate gates are only reachable
if the design has somewhere to spread to. A single workspace with
thirty crates would satisfy no gate and would give a reader no map.

The payoff compounds. Splitting one fat crate into two sibling crates
can clear file-LOC, module-function-count and crate-module-count
warnings simultaneously.

## 3. Facades

`lib.rs` and `mod.rs` are facades ONLY. No executable logic -- `pub
use` re-exports and module declarations. The same rule applies to
`build.rs`. Behavior lives in named files.

## 4. File-name convention

Agents and humans both read these names to know where new code belongs.
Use them.

| File | Holds |
| --- | --- |
| `model.rs` | data types, no behavior |
| `parse.rs` | input -> typed |
| `validate.rs` | typed -> result |
| `plan.rs` | config -> plan |
| `run.rs` | effects |
| `render.rs` | data -> string |
| `error.rs` | error types |
| `test_support.rs` | helpers used by tests |
| `fixtures.rs` | test data |

## 5. How to split

**Split by responsibility, never mechanically.** When a function
exceeds 25 LOC, the useful question is not "where is the midpoint" but
"what TWO jobs is this doing". Extract the one that has a name.

Three lenses, in the order worth trying:

1. **Phase separation.** Code runs at one of four phases: compile time,
   start-up, conditional, or dataflow. An over-budget function is
   almost always mixing phases, and splitting by phase retires the
   warning naturally. A fifth phase -- pre-compile via `build.rs` --
   lets external data drive codegen. The Gowin device profile table in
   `components/device/` is a likely `build.rs` candidate: it is
   external data that several crates need in several shapes.

2. **Compose, do not compress.** Top-down delegation, iterator
   pipelines, `?`-chained `Result`/`Option`, builder and visitor
   patterns. Functional style is compact style, and it stays readable
   at 25 LOC in a way that a densely packed imperative block does not.

3. **Define once, invoke many.** Macros (per-site cost: one line),
   trait blanket impls, `const` lookup tables, `build.rs` codegen.
   Drive boilerplate from a single source of truth across
   initialization, dispatch, docs and domain logic.

Further directives:

- **Move tests out of production modules** when an inline
  `#[cfg(test)] mod tests { ... }` block distorts readability. Use a
  sibling `parse_tests.rs` next to `parse.rs`, or a `tests/` dir.
  Test code still counts toward file LOC.
- **Prefer pure free functions over methods on `Self`** for parsing,
  validating, transforming and classifying. Reserve `impl` methods for
  constructors, state mutation, and invariant-preserving operations.
- **Separate decisions from effects.** Pure code decides; a thin shell
  performs IO and mutation. This matters more here than in most
  projects: the whole point of the emulator is that a decision made in
  software must match a decision made in silicon, and pure decision
  functions are the ones that can be golden-tested against RTL.
- **Do not add new logic to an over-limit function or module.** Extract
  responsibilities into named pure helpers first, then add.

## 6. Symptom to technique

| Symptom | First refactor to try |
| --- | --- |
| Function LOC over budget | phase separation; extract the named sub-job |
| Module function count over budget | split by responsibility into a sibling module |
| File LOC over budget | move tests out; then split the module |
| Crate module count over budget | promote a cluster of modules to a sibling crate |
| Component crate count over budget | promote a crate cluster to a new component |
| Clippy allows accumulating | fix the cause; an `#[allow]` is a deferred failure |

## 7. Ratchet policy

Every commit should strictly lower BOTH the `sw-checklist` failed count
and the warning count from its parent. Holding steady is not enough.

- Run `sw-checklist` before committing; note the counts.
- If your commit introduces a new FAIL, retire it before shipping.
- Record before/after counts in a `sw-checklist:` commit trailer.
- A commit that grows or merely holds the counts MUST carry
  `sw-checklist: exception` on its own line, saying what was tried and
  why retirement was not feasible in that step. Use sparingly.

This repo starts at zero code, so the honest goal is different from a
paydown project: **keep both counts at zero from the first crate
onward.** It is far cheaper to never acquire the debt than to retire
it later, and there is no legacy here to excuse a nonzero baseline.
