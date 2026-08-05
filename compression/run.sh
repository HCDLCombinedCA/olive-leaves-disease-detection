#!/usr/bin/env bash
# Run a command inside the course image. No extra packages are installed.
#
#   ./run.sh python src/prepare_data.py
#   ./run.sh bash
#
# We deliberately use kquille/hcaim_gpu_tudublin_course:latest unmodified, with
# no pip install step, so that everything here stays inside the toolset the
# course labs provide:
#   * quantisation  -> tf.lite (TensorFlow 2.12 core, not an add-on)
#   * pruning       -> Keras get_weights/set_weights (Weeks 1-4)
#   * model export  -> SavedModel (Week 7 Production)
#   * XAI           -> lime / shap (Week 8 XAI), tf-keras-vis
#
# compression/ is mounted at /work, so relative paths in the scripts behave the
# same inside and outside the container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
IMAGE="${IMAGE:-kquille/hcaim_gpu_tudublin_course:latest}"

# Only pass -it when actually attached to a terminal; otherwise this fails in
# non-interactive contexts such as CI or background jobs.
TTY_FLAGS=""
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

exec docker run --rm $TTY_FLAGS \
    --user "$(id -u):$(id -g)" \
    -v "$HERE:/work" \
    -v "$REPO/olive-leaf-image-dataset.zip:/data/olive-leaf-image-dataset.zip:ro" \
    -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -w /work \
    "$IMAGE" "$@"
