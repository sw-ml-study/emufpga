#!/usr/bin/env bash
set -euo pipefail

# Run a build/test command under a GLOBAL lock so two cargo invocations
# can NEVER run at the same time.
#
# Why this exists: every component workspace in this repo shares one
# target/ dir (.cargo/config.toml). Two concurrent cargo runs block on
# the same build lock and DEADLOCK -- producing no output while
# wall-clock burns. Routing every cargo call through this wrapper
# serializes them: a second caller WAITS instead of deadlocking.
#
#   scripts/serial.sh cargo test -p spm-codec
#
# The lock is held only for the duration of THIS command, not across
# nested script calls, so a script may call serial.sh many times in
# sequence without self-deadlocking.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/target"
LOCK="$ROOT/target/.build.lock"
WAIT_SECS="${SERIAL_WAIT_SECS:-1800}"

# Preferred path (Linux): flock waits up to WAIT_SECS, then fails
# loudly rather than hanging forever, and auto-releases if the holder
# dies. exec replaces the shell so signals and exit codes pass through.
if command -v flock >/dev/null 2>&1; then
    exec flock -w "$WAIT_SECS" "$LOCK" "$@"
fi

# Portable fallback for hosts without flock (notably macOS, which does
# not ship it): an atomic-mkdir spinlock. mkdir is atomic on POSIX
# filesystems, so a directory serves as the mutex. We record the holder
# PID so a lock orphaned by a SIGKILLed build (its EXIT trap never ran)
# can be reclaimed instead of stalling the full timeout.
LOCKDIR="$LOCK.d"
deadline=$(( $(date +%s) + WAIT_SECS ))
while ! mkdir "$LOCKDIR" 2>/dev/null; do
    holder="$(cat "$LOCKDIR/pid" 2>/dev/null || true)"
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
        rm -rf "$LOCKDIR" # stale: the recorded holder is gone
        continue
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "serial.sh: timed out after ${WAIT_SECS}s waiting for $LOCKDIR" >&2
        exit 1
    fi
    sleep 1
done
trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
echo $$ > "$LOCKDIR/pid"
"$@"
