"""Compare model-predicted leaf counts against ground-truth counts.

Reports the standard CVPPP Leaf Counting Challenge (LCC) metrics:

    DiC     mean signed count difference  (pred - gt)   -> counting *bias*
    |DiC|   mean absolute count difference              -> error magnitude
    MSE     mean squared error
    %agree  fraction of images counted exactly right (pred == gt)

(std of DiC and |DiC| are shown in parentheses, as on the CVPPP leaderboard.)

Ground truth: the CVPPP `A?.csv` files (rows `image, count`, no header) — or a
folder containing them, in which case every `*.csv` under it is merged. Any CSV
of `image,count` works.

Predictions come from one of two sources:
  --pred FILE       a CSV of `image,n_leaves`, as written by a run script's
                    `--count-csv` flag
  --from-crops DIR  a `--crops` output dir; the predicted count for an image is
                    the number of `leaf_*.png` in its `<image>_leaves/` folder
                    (evaluate a --crops run without re-running the model)

Rows are matched by image name ignoring the extension (`plant001_rgb.png`, a
`.jpg`, and the crop folder `plant001_rgb_leaves` all match), so predictions and
ground truth may live in different folders. Pure stdlib — runs in any venv.

Examples:
    # from a --count-csv predictions file
    python eval/evaluate_leaf_count.py \
        --pred outputs/A1/counts.csv \
        --gt data/samples/A1/A1.csv

    # straight from a --crops run's output folder, with a full table
    python eval/evaluate_leaf_count.py \
        --from-crops 2d_foundation/01_sam2/outputs/A1 \
        --gt data/samples/A1/A1.csv \
        --per-image --out results.csv
"""
import argparse
import csv
import glob
import math
import os
import sys


def _to_int(s):
    """Parse a count field ('15', '  5', '5.0') to int, or None if not numeric."""
    s = str(s).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(round(float(s)))
        except ValueError:
            return None


def _key(name):
    """Normalize an image reference to a match key: basename without extension
    (so `plant001_rgb.png`, a `.jpg`, and a crop folder `plant001_rgb_leaves`
    all line up)."""
    return os.path.splitext(os.path.basename(str(name).strip()))[0]


def read_counts(path):
    """Read an `image, count` CSV into {key: count} keyed by `_key(image)`.

    Header-agnostic: a first row whose second column is non-numeric is skipped.
    Whitespace is stripped.
    """
    counts = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name = row[0].strip()
            n = _to_int(row[1])
            if not name or n is None:
                continue  # header row or malformed line
            counts[_key(name)] = n
    return counts


def counts_from_crops(root):
    """Derive predicted counts from a `--crops` output dir: for each
    `<stem>_leaves` subfolder, count its `leaf_*.png` cutouts. Lets you evaluate
    a `--crops` run without re-running the model."""
    dirs = [d for d in glob.glob(os.path.join(root, "**", "*_leaves"), recursive=True)
            if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"No '*_leaves' crop folders found under {root} "
                 "(run a model with --crops, or use --pred with a --count-csv file).")
    counts = {}
    for d in sorted(dirs):
        stem = os.path.basename(d)[:-len("_leaves")]
        counts[_key(stem)] = len(glob.glob(os.path.join(d, "leaf_*.png")))
    return counts


def gather_gt(path):
    """Return (counts, csv_files, collisions) from a CSV file or folder of CSVs.

    `collisions` counts basenames that appear in more than one CSV with
    *conflicting* counts (the CVPPP A1/A2/A3 subsets all reuse `plantNNN_rgb.png`
    names) — merging by basename is then ambiguous, so the caller warns.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        if not files:
            sys.exit(f"No .csv ground-truth files found under {path}")
    else:
        files = [path]
    merged = {}
    collisions = 0
    for fp in files:
        for name, n in read_counts(fp).items():
            if name in merged and merged[name] != n:
                collisions += 1
            merged[name] = n
    return merged, files, collisions


def evaluate(pred, gt):
    """Compute per-image diffs (pred - gt) over the shared image basenames."""
    shared = [k for k in pred if k in gt]
    rows = [(k, gt[k], pred[k], pred[k] - gt[k]) for k in sorted(shared)]
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    diffs = [d for *_, d in rows]
    absd = [abs(d) for d in diffs]
    dic = sum(diffs) / n
    absdic = sum(absd) / n
    mse = sum(d * d for d in diffs) / n
    agree = 100.0 * sum(1 for d in diffs if d == 0) / n
    dic_std = math.sqrt(sum((d - dic) ** 2 for d in diffs) / n)
    absdic_std = math.sqrt(sum((a - absdic) ** 2 for a in absd) / n)
    return {
        "n": n, "DiC": dic, "DiC_std": dic_std,
        "absDiC": absdic, "absDiC_std": absdic_std,
        "MSE": mse, "RMSE": math.sqrt(mse), "agree": agree,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred",
                     help="predictions CSV (image,n_leaves) from a run's --count-csv")
    src.add_argument("--from-crops",
                     help="a --crops output dir: predicted count = number of "
                          "leaf_*.png in each <image>_leaves/ subfolder")
    ap.add_argument("--gt", required=True,
                    help="ground-truth CSV, or a folder of CVPPP A?.csv files")
    ap.add_argument("--per-image", action="store_true",
                    help="print the full per-image gt/pred/diff table")
    ap.add_argument("--out", help="write per-image results to this CSV")
    args = ap.parse_args()

    if args.from_crops:
        pred = counts_from_crops(args.from_crops)
        pred_src = args.from_crops + "  (leaf_*.png per *_leaves folder)"
    else:
        pred = read_counts(args.pred)
        pred_src = args.pred
    gt, gt_files, collisions = gather_gt(args.gt)
    rows = evaluate(pred, gt)

    if args.per_image and rows:
        print(f"{'image':<24}{'gt':>5}{'pred':>6}{'diff':>6}")
        print("-" * 41)
        for name, g, p, d in rows:
            print(f"{name:<24}{g:>5}{p:>6}{d:>+6}")
        print()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "gt", "pred", "diff"])
            w.writerows(rows)
        print(f"Wrote per-image results -> {args.out}")

    stats = summarize(rows)
    only_pred = sorted(set(pred) - set(gt))
    only_gt = sorted(set(gt) - set(pred))

    print(f"predictions : {pred_src}  ({len(pred)} images)")
    print(f"ground truth: {args.gt}  ({len(gt)} images, {len(gt_files)} csv)")
    if collisions:
        print(f"  WARNING: {collisions} image name(s) appear in multiple GT csvs "
              "with different counts (A1/A2/A3 reuse names). Merged counts are "
              "ambiguous — evaluate one subset at a time (--gt A1/A1.csv).")
    print(f"matched     : {len(rows)} images")
    if only_pred:
        print(f"  {len(only_pred)} predicted image(s) not in ground truth"
              f" (e.g. {only_pred[:3]})")
    if only_gt:
        print(f"  {len(only_gt)} ground-truth image(s) not predicted"
              f" (e.g. {only_gt[:3]})")

    if not stats:
        sys.exit("\nNo overlapping images between predictions and ground truth "
                 "— check that image basenames match.")

    print("\n=== CVPPP leaf-counting metrics ===")
    print(f"  DiC   (bias)      : {stats['DiC']:+.3f}  (std {stats['DiC_std']:.3f})")
    print(f"  |DiC| (abs error) : {stats['absDiC']:.3f}  (std {stats['absDiC_std']:.3f})")
    print(f"  MSE               : {stats['MSE']:.3f}   (RMSE {stats['RMSE']:.3f})")
    print(f"  Agreement         : {stats['agree']:.1f}%  "
          f"({round(stats['agree'] / 100 * stats['n'])}/{stats['n']} exact)")


if __name__ == "__main__":
    main()
