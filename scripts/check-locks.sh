#!/usr/bin/env bash
set -euo pipefail

# Repo-wide Cargo.lock consistency sweep.
#
# Every component directory is its own Cargo workspace with its own
# Cargo.lock, and a lock also pins path-dependencies OWNED BY OTHER
# workspaces. So a manifest change in one component silently strands
# the locks of every downstream workspace until someone builds there.
# This script fails fast on any stale lock.
#
#   scripts/check-locks.sh          # verify only (CI-safe)
#   scripts/check-locks.sh --fix    # regenerate stale locks in place
#
# Components that do not yet contain a Cargo.toml are skipped: the
# directory is a placeholder for a workspace a later saga fills in.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="${1:-}"
stale=0
checked=0

for ws in "$ROOT"/components/*/; do
    [ -f "$ws/Cargo.toml" ] || continue
    checked=$((checked + 1))
    if ! (cd "$ws" && cargo metadata --locked --format-version 1 >/dev/null 2>&1); then
        name="$(basename "$ws")"
        if [ "$FIX" = "--fix" ]; then
            echo "regenerating stale lock: $name"
            (cd "$ws" && "$ROOT/scripts/serial.sh" cargo metadata --format-version 1 >/dev/null)
        else
            echo "STALE LOCK: components/$name (run scripts/check-locks.sh --fix)"
            stale=1
        fi
    fi
done

if [ "$stale" -ne 0 ]; then
    exit 1
fi

if [ "$checked" -eq 0 ]; then
    echo "no component workspaces yet -- nothing to check"
else
    echo "all $checked workspace lock(s) consistent"
fi
