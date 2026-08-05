#!/usr/bin/env bash
# End-to-end compression pipeline, reproducible from a clean checkout.
#
#   ./pipeline.sh                 # MobileNetV2 (primary target)
#   ./pipeline.sh densenet121     # heavier comparison backbone
#
# Every step runs inside the unmodified course image via run.sh. Steps are
# ordered by dependency: the dataset must be de-leaked before anything is
# trained, and the baseline must exist before it can be compressed.
#
# Run it once per backbone, then `python3 ../make_report_tables.py -o ../RESULTS.md`
# to regenerate the combined report tables.
#
# Expect roughly 1 hour for mobilenetv2 and 3 hours for densenet121 on a CPU-only
# machine; the sparsity sweep is the bulk of that.
set -euo pipefail

BACKBONE="${1:-mobilenetv2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Sparsity levels differ per backbone because the two behave nothing alike.
# MobileNetV2 is swept low: it is built from depthwise separable convolutions,
# has little redundancy to give up, and already loses ground by 35%. DenseNet121
# is swept high: 50% costs it nothing at all, so the question is where its
# redundancy finally runs out.
if [ "$BACKBONE" = "densenet121" ]; then
    SPARSITIES="0.5 0.65 0.8 0.9"
else
    SPARSITIES="0.2 0.35 0.5 0.65"
fi

echo "### 1/7  dataset: extract, de-leak, split"
if [ ! -f data/prepared/stats.json ]; then
    ./run.sh python src/prepare_data.py
else
    echo "     data/prepared already present, skipping"
fi

echo "### 2/7  baseline: $BACKBONE"
if [ ! -d "artifacts/${BACKBONE}_baseline" ]; then
    ./run.sh python src/train_baseline.py --backbone "$BACKBONE"
else
    echo "     artifacts/${BACKBONE}_baseline already present, skipping"
fi

echo "### 3/7  quantisation variants"
./run.sh python src/quantise.py --backbone "$BACKBONE"

echo "### 4/7  magnitude pruning at 50% + fine-tune"
./run.sh python src/prune.py --backbone "$BACKBONE" --sparsity 0.5

echo "### 5/7  pruning stacked with quantisation"
./run.sh python src/quantise.py --backbone "$BACKBONE" --tag pruned50 \
    --variants dynamic_range float16

echo "### 6/7  sparsity sweep ($SPARSITIES)"
# shellcheck disable=SC2086
./run.sh python src/prune_sweep.py --backbone "$BACKBONE" --sparsities $SPARSITIES

echo "### 7/7  measurement, figures, explanation fidelity"
# Measurement must not share the machine with anything else: latency is one of
# the headline metrics and any competing load inflates it.
./run.sh python src/measure.py --backbone "$BACKBONE"
./run.sh python src/xai_fidelity.py --backbone "$BACKBONE"
./run.sh python src/plot_results.py

echo
echo "Done. Results in results/:"
ls -1 results/
