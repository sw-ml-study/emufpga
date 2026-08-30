#!/usr/bin/env bash
set -euo pipefail

# The pre-commit gate for ONE component workspace, run serially.
#
#   scripts/gate.sh <workspace-dir> [pkg...]
#
# <workspace-dir> is the component workspace that owns the packages
# (this repo has no root Cargo.toml, so `cargo -p` must run inside the
# owning workspace, e.g. components/format). With no packages named,
# the whole workspace is gated.
#
# Order is deliberate: lock consistency first (a stale lock means a
# manifest changed without regeneration, and every later step would be
# testing the wrong dependency versions), then formatting, then lint,
# then tests, then project standards.
#
# See docs/code_metrics.md for the complexity gates sw-checklist
# enforces and how to refactor when a check fails.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERIAL="$SCRIPT_DIR/serial.sh"

if [ "$#" -lt 1 ]; then
    echo "usage: scripts/gate.sh <workspace-dir> [pkg...]" >&2
    exit 2
fi

WS="$1"; shift

if [ ! -f "$WS/Cargo.toml" ]; then
    echo "gate.sh: $WS has no Cargo.toml -- nothing to gate" >&2
    exit 2
fi

# No packages named: gate the whole workspace. Named packages: scope to
# them, per the checkpoint rule "scope tests to what you changed".
#
# fmt and clippy/test disagree on the spelling of "everything":
# `cargo fmt` takes --all, while clippy and test take --workspace. Keep
# two arrays rather than one, or the whole-workspace path breaks.
if [ "$#" -eq 0 ]; then
    FMT_SCOPE=(--all)
    CARGO_SCOPE=(--workspace)
    LABEL="(workspace)"
else
    FMT_SCOPE=()
    CARGO_SCOPE=()
    for p in "$@"; do
        FMT_SCOPE+=(-p "$p")
        CARGO_SCOPE+=(-p "$p")
    done
    LABEL="($*)"
fi

cd "$WS"

echo "=== Cargo.lock consistency: $WS ==="
# A stale lock here means a manifest changed -- possibly in ANOTHER
# workspace this one path-depends on -- without the lock being
# regenerated. Fix with scripts/check-locks.sh --fix.
"$SERIAL" cargo metadata --locked --format-version 1 >/dev/null

echo "=== rustfmt --check $LABEL ==="
"$SERIAL" cargo fmt "${FMT_SCOPE[@]}" -- --check

echo "=== clippy -D warnings $LABEL ==="
"$SERIAL" cargo clippy "${CARGO_SCOPE[@]}" --all-targets --all-features -- -D warnings

echo "=== cargo test $LABEL ==="
"$SERIAL" cargo test "${CARGO_SCOPE[@]}"

echo "=== gate OK: $WS $LABEL ==="
