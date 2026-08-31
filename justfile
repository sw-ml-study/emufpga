set shell := ["sh", "-cu"]

# This justfile ONLY delegates. Every build entry point lives in
# ./scripts so that the same commands work from CI, from an agent
# session, and from a shell without just installed.

default:
    @just --list

# Pre-commit gate over the components you changed.
check:
    ./scripts/check

# Pre-commit gate over every component workspace.
check-all:
    ./scripts/check --all

# Gate one component: `just gate format`
gate component:
    ./scripts/gate.sh components/{{component}}

# Build every component workspace, serially.
build:
    ./scripts/build

build-release:
    ./scripts/build --release

# Reproduce the docs/results.md measurement.
bench:
    ./scripts/bench

# Repo-wide Cargo.lock consistency sweep.
locks:
    ./scripts/check-locks.sh

locks-fix:
    ./scripts/check-locks.sh --fix

# Project standards (complexity gates -- see docs/code_metrics.md).
checklist:
    sw-checklist

# Markdown must be ASCII-only.
markdown:
    sw-markdown-checker -f "**/*.md"
