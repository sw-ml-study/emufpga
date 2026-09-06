#!/usr/bin/env bash
set -euo pipefail

# Print the target CUDA compute architecture as a packed integer:
# compute capability 8.6 (Ampere, RTX 3060) -> 86, 12.0 (Blackwell,
# RTX 5060) -> 120. Override detection with CUDA_ARCH (a trailing "a"
# accelerated suffix is stripped for this dir/tag purpose).
#
# The GPU experiment scripts were first built on an RTX 5060 (sm_120a);
# this keeps them portable to whatever card is present without editing a
# hardcoded flag.

if [ -n "${CUDA_ARCH:-}" ]; then
    printf '%s\n' "${CUDA_ARCH%a}"
    exit 0
fi

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
[ -n "$cap" ] || { echo "cannot detect CUDA arch; set CUDA_ARCH (e.g. 86 or 120)" >&2; exit 2; }
printf '%s\n' "$cap"
