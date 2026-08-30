Scaffold the emufpga repository so that every later step has a working
build and gate. Read docs/plan.md sections 5 and 9 first -- they are the
specification for this step. Model the layout on ../sw-mlpl.

Deliverables:

1. `components/` directory with one subdirectory per component named in
   docs/plan.md section 5 (format, stream, tensor, device, fabric,
   engine, report, cli). Each is its OWN cargo workspace with its own
   Cargo.toml and Cargo.lock -- there is NO root Cargo.toml. Only the
   components saga 1 builds need a real workspace manifest now; the
   rest may be empty directories with a .gitkeep and a one-line README
   naming the saga that fills them.

2. `.cargo/config.toml` pointing every workspace at one shared
   `target/` dir.

3. `scripts/` -- all build entry points live here, per docs/plan.md
   section 9:
   - `serial.sh` -- global build lock. Port ../sw-mlpl/scripts/serial.sh
     (flock on Linux, atomic-mkdir spinlock fallback on macOS). This is
     mandatory: all components share one target/, so two concurrent
     cargo invocations deadlock on the same build lock.
   - `gate.sh <workspace-dir> <pkg>...` -- cargo metadata --locked,
     cargo fmt -- --check, cargo clippy --all-targets --all-features
     -- -D warnings, cargo test, then sw-checklist. Every cargo call
     routed through serial.sh.
   - `check-locks.sh [--fix]` -- repo-wide Cargo.lock consistency
     sweep across components/*/.
   - `check` -- run gate.sh over the components that changed.
   - `build` -- build all components serially.

4. `justfile` that ONLY delegates to scripts/ (default recipe lists
   targets; check, build, test, fmt, clippy, fit recipes).

5. Workspace conventions in every component Cargo.toml:
   edition 2024, `[workspace.package]` with license and version, and
   `[workspace.lints.clippy] pedantic = "warn"`.

6. `.gitignore` -- target/, .DS_Store, editor noise, and
   `.agentrail/sessions/` (session transcripts stay local; steps and
   trajectories stay tracked).

7. `LICENSE` (MIT OR Apache-2.0), `COPYRIGHT`, and a `README.md` that
   states what emufpga is and is not (docs/plan.md section 1), names
   the Tang Nano 9K as the primary target, and links docs/plan.md and
   docs/research.txt.

8. `docs/code_metrics.md` -- adapt ../sw-mlpl's complexity gate policy:
   25 LOC/function, 5 functions/module, 5 modules/crate, 5
   crates/component; lib.rs and mod.rs are facades only; the file-name
   convention (parse.rs, validate.rs, plan.rs, run.rs, render.rs,
   error.rs, model.rs, test_support.rs, fixtures.rs); split by
   responsibility, never with whitespace tricks or #[allow].

9. Extend CLAUDE.md BELOW the agentrail markered block (never edit
   inside the markers) with: the scripts/ build discipline, the
   ASCII-only markdown rule, the sw-checklist ratchet policy, and a
   pointer to docs/code_metrics.md.

10. Record the baseline `sw-checklist` counts in the commit message as
    a `sw-checklist:` trailer, so the ratchet has a starting point.

Acceptance: `just check` runs green on the otherwise-empty tree,
`scripts/check-locks.sh` passes, `sw-markdown-checker -f "**/*.md"`
passes on every hand-written .md (CLAUDE.md's agentrail block is a
known upstream failure -- exclude it, do not edit it), and
`sw-checklist` reports a recorded baseline.

Do NOT write any SPM format, stream, or tensor code in this step. That
is step 2 onward.
