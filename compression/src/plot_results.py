"""Report figures for the compression results.

Two figures, each answering one question:

1. `pruning_sweep.png` -- how far can sparsity go before accuracy falls away, and
   does the compact backbone tolerate it as well as the heavy one?
2. `compression_tradeoff.png` -- size against accuracy, and latency against
   accuracy, side by side. Two panels rather than one chart with two x-scales,
   because the interesting result is precisely that the two disagree: the
   smallest variant is not the fastest.

Colour encodes the backbone only (two slots from the reference categorical
palette, validated all-pairs: worst CVD dE 24.7, normal-vision dE 33.6). The
compression variant is encoded by marker shape, which keeps the colour count at
two and leaves identity legible without relying on hue alone.

Usage:
    ./run.sh python src/plot_results.py
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette, light mode.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"mobilenetv2": "#2a78d6", "densenet121": "#eb6834"}

# Marker shape carries the variant, so identity never rests on colour alone.
MARKERS = {
    "baseline": "o",
    "float32": "o",
    "float16": "s",
    "dynamic_range": "^",
    "full_integer": "D",
}
LABELS = {
    "float32": "fp32",
    "float16": "fp16",
    "dynamic_range": "dyn-int8",
    "full_integer": "int8",
    "baseline": "Keras",
}


def style_axes(axis, xlabel, ylabel, title):
    """Recessive grid and axes; text in ink tokens, never in a series colour."""
    axis.set_facecolor(SURFACE)
    axis.set_axisbelow(True)
    axis.grid(True, color=GRIDLINE, linewidth=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE)
        axis.spines[side].set_linewidth(1.0)
    axis.tick_params(colors=INK_MUTED, labelsize=9)
    axis.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    axis.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    axis.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)


def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_baseline_f1(backbone):
    import json
    path = os.path.join("results", "%s_baseline.json" % backbone)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)["metrics"]["test"]["macro_f1"]


def plot_sweep(out_dir):
    figure, axis = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    plotted = False
    present = []

    for backbone, colour in SERIES.items():
        rows = read_csv(os.path.join("results", "%s_pruning_sweep.csv" % backbone))
        if not rows:
            continue
        plotted = True
        present.append(backbone)
        xs = [0.0]
        ys = [read_baseline_f1(backbone) or float("nan")]
        for row in sorted(rows, key=lambda r: float(r["target_sparsity"])):
            xs.append(float(row["target_sparsity"]) * 100)
            ys.append(float(row["test_macro_f1"]))
        axis.plot(xs, ys, color=colour, linewidth=2.0, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=backbone, zorder=3)
        # Direct label at the end of the line, so identity is not colour-alone.
        axis.annotate(backbone, xy=(xs[-1], ys[-1]), xytext=(6, 0),
                      textcoords="offset points", color=INK_SECONDARY,
                      fontsize=9, va="center")

    if not plotted:
        plt.close(figure)
        return None

    title = "Accuracy against pruning sparsity, after recovery fine-tuning"
    if len(present) == 1:
        # One series needs no legend box; the title names it instead.
        title = "%s: accuracy against pruning sparsity, after recovery" % present[0]
    else:
        axis.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower left")
    style_axes(axis, "Sparsity of prunable kernels (%)", "Test macro-F1", title)
    # Headroom on the right so the end-of-line labels are not clipped.
    left, right = axis.get_xlim()
    axis.set_xlim(left, right + (right - left) * 0.16)
    figure.tight_layout()
    path = os.path.join(out_dir, "pruning_sweep.png")
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_tradeoff(out_dir):
    # Shared y so the two panels can be read against each other: the whole point
    # is that a variant's position differs between them.
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), facecolor=SURFACE, sharey=True)
    plotted = False
    present = []

    for backbone, colour in SERIES.items():
        rows = read_csv(os.path.join("results", "%s_compression_table.csv" % backbone))
        if not rows:
            continue
        present.append(backbone)
        for row in rows:
            name = row["variant"].replace("%s_" % backbone, "")
            # The sweep and pruned models crowd the plot without adding to the
            # size/latency story; the quantisation family is the point here.
            if "pruned" in name or "sweep" in name:
                continue
            variant = name.replace("baseline_", "") if name != "baseline" else "baseline"
            marker = MARKERS.get(variant, "o")
            f1 = float(row["test_macro_f1"])
            plotted = True
            for axis, key in ((axes[0], "size_mb"), (axes[1], "latency_median_ms")):
                axis.scatter(float(row[key]), f1, s=90, color=colour, marker=marker,
                             edgecolor=SURFACE, linewidth=2, zorder=3)
                axis.annotate(LABELS.get(variant, variant),
                              xy=(float(row[key]), f1), xytext=(7, -3),
                              textcoords="offset points", color=INK_SECONDARY, fontsize=8)

    if not plotted:
        plt.close(figure)
        return None

    style_axes(axes[0], "Model size (MB)", "Test macro-F1", "Smaller is not free")
    style_axes(axes[1], "CPU latency, median (ms)", "Test macro-F1",
               "…and smallest is not fastest")

    # Only list backbones that actually have points, so the legend never
    # advertises a series the reader cannot find.
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          color=SERIES[backbone], markeredgecolor=SURFACE,
                          markeredgewidth=2, label=backbone)
               for backbone in present]
    axes[0].legend(handles=handles, frameon=False, labelcolor=INK_SECONDARY,
                   fontsize=9, loc="lower right")
    figure.tight_layout()
    path = os.path.join(out_dir, "compression_tradeoff.png")
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name, function in (("pruning sweep", plot_sweep),
                           ("compression trade-off", plot_tradeoff)):
        path = function(args.out)
        if path:
            print("wrote %s" % path)
        else:
            print("skipped %s: inputs not available yet" % name)


if __name__ == "__main__":
    main()
