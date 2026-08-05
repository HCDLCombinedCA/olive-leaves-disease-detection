#!/usr/bin/env bash
# Run a command inside the unmodified course image, with the repository root
# mounted so scripts can read the shared datasets produced by other components.
#
#   ./run.sh python glassbox_fixed.py
#
# No pip install anywhere: the image is used exactly as delivered.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
IMAGE="${IMAGE:-kquille/hcaim_gpu_tudublin_course:latest}"

TTY_FLAGS=""
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

exec docker run --rm $TTY_FLAGS \
    --user "$(id -u):$(id -g)" \
    -v "$REPO:/repo" \
    -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -w /repo/analysis \
    "$IMAGE" "$@"
