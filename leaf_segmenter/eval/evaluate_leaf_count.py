"""Compare model-predicted leaf counts against ground-truth counts (CVPPP LCC).

Predictions are read from `<input-dir>/counts.csv` (image,n_leaves — the file any
run script writes when given --output-dir). Ground truth is read from the CVPPP
`A?.csv` file(s) under --gt-dir (rows `image,count`). Rows are matched by image
basename ignoring the extension, so the two folders can differ.

Reports the standard CVPPP Leaf Counting Challenge metrics:
    DiC     mean signed count difference (pred - gt)  -> counting bias
    |DiC|   mean absolute count difference            -> error magnitude
    MSE     mean squared error (RMSE alongside)
    %agree  fraction of images counted exactly right

With --output-dir, also writes the full per-image gt/pred/diff table there. Pure
stdlib — runs in any venv.

Examples:
    python eval/evaluate_leaf_count.py \
        --input-dir 2d_foundation/01_sam2/output/small/A1 \
        --gt-dir data/cvppp/images/A1
    python eval/evaluate_leaf_count.py \
        --input-dir out/A1 --gt-dir data/cvppp/images/A1 --output-dir eval_out/A1
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
    """Match key: basename without extension (so `plant001_rgb.png` and a `.jpg`
    line up)."""
    return os.path.splitext(os.path.basename(str(name).strip()))[0]


def read_counts(path):
    """Read an `image, count` CSV into {key: count}. Header-agnostic: a first row
    whose second column is non-numeric is skipped."""
    counts = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name, n = row[0].strip(), _to_int(row[1])
            if name and n is not None:
                counts[_key(name)] = n
    return counts


def read_gt(gt_dir):
    """Merge every *.csv under gt_dir into {key: count}.

    `collisions` counts basenames that appear in more than one CSV with different
    counts — the CVPPP A1/A2/A3 subsets reuse `plantNNN_rgb.png` names, so merging
    across them is ambiguous. Evaluate one subset at a time.
    """
    files = sorted(glob.glob(os.path.join(gt_dir, "**", "*.csv"), recursive=True))
    if not files:
        sys.exit(f"No .csv ground-truth files found under {gt_dir}")
    merged, collisions = {}, 0
    for fp in files:
        for name, n in read_counts(fp).items():
            if name in merged and merged[name] != n:
                collisions += 1
            merged[name] = n
    return merged, files, collisions


def evaluate(pred, gt):
    """Per-image rows (image, gt, pred, pred-gt) over the shared basenames."""
    return [(k, gt[k], pred[k], pred[k] - gt[k]) for k in sorted(pred) if k in gt]


def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    diffs = [d for *_, d in rows]
    absd = [abs(d) for d in diffs]
    dic, absdic = sum(diffs) / n, sum(absd) / n
    mse = sum(d * d for d in diffs) / n
    return {
        "n": n,
        "DiC": dic, "DiC_std": math.sqrt(sum((d - dic) ** 2 for d in diffs) / n),
        "absDiC": absdic, "absDiC_std": math.sqrt(sum((a - absdic) ** 2 for a in absd) / n),
        "MSE": mse, "RMSE": math.sqrt(mse),
        "agree": 100.0 * sum(1 for d in diffs if d == 0) / n,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="model output dir containing counts.csv (image,n_leaves)")
    ap.add_argument("--gt-dir", required=True,
                    help="folder with the CVPPP A?.csv ground-truth file(s)")
    ap.add_argument("--output-dir",
                    help="if given, write the full per-image gt/pred/diff table here")
    args = ap.parse_args()

    pred_csv = os.path.join(args.input_dir, "counts.csv")
    if not os.path.isfile(pred_csv):
        sys.exit(f"No counts.csv in {args.input_dir} "
                 "(run a model with --output-dir first).")
    pred = read_counts(pred_csv)
    gt, gt_files, collisions = read_gt(args.gt_dir)
    rows = evaluate(pred, gt)

    print(f"predictions : {pred_csv}  ({len(pred)} images)")
    print(f"ground truth: {args.gt_dir}  ({len(gt)} images, {len(gt_files)} csv)")
    if collisions:
        print(f"  WARNING: {collisions} image name(s) appear in multiple GT csvs "
              "with different counts (A1/A2/A3 reuse names) — evaluate one subset "
              "at a time.")
    print(f"matched     : {len(rows)} images")

    stats = summarize(rows)
    if not stats:
        sys.exit("\nNo overlapping images between predictions and ground truth "
                 "— check that image basenames match.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "leaf_count_per_image.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "gt", "pred", "diff"])
            w.writerows(rows)
        print(f"\n{'image':<24}{'gt':>5}{'pred':>6}{'diff':>6}")
        print("-" * 41)
        for name, g, p, d in rows:
            print(f"{name:<24}{g:>5}{p:>6}{d:>+6}")
        print(f"\nWrote per-image table -> {out}")

    print("\n=== CVPPP leaf-counting metrics ===")
    print(f"  DiC   (bias)      : {stats['DiC']:+.3f}  (std {stats['DiC_std']:.3f})")
    print(f"  |DiC| (abs error) : {stats['absDiC']:.3f}  (std {stats['absDiC_std']:.3f})")
    print(f"  MSE               : {stats['MSE']:.3f}   (RMSE {stats['RMSE']:.3f})")
    print(f"  Agreement         : {stats['agree']:.1f}%  "
          f"({round(stats['agree'] / 100 * stats['n'])}/{stats['n']} exact)")


if __name__ == "__main__":
    main()
