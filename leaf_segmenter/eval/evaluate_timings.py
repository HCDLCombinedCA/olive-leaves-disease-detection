"""Summarize per-model runtime from the timings.csv each run script writes.

Every run script given --output-dir writes `timings.csv`
(image,inference_s,postprocess_s — one row per image, via
shared.helper.save_timings_csv). This evaluator reads one or more of those model
output dirs and reports, per model, the mean, median, and max of the inference
time, the post-processing time, and their total. Pure stdlib — runs in any venv.

Note: the first image of a run usually includes one-off warm-up cost (CUDA
kernel loading etc.), which inflates the mean more than the median.

With --output-dir, also writes the summary table as timing_summary.csv.

Examples:
    python eval/evaluate_timings.py --input-dir 2d_foundation/01_sam2/out/A1
    python eval/evaluate_timings.py \
        --input-dir 2d_foundation/01_sam2/out/A1 2d_foundation/03_hq_sam/out/A1 \
        --output-dir eval_out/A1
"""
import argparse
import csv
import os
import sys
from statistics import mean, median


def read_timings(path):
    """Read a timings.csv into [(image, inference_s, postprocess_s)].
    Header-agnostic: rows whose time columns are non-numeric are skipped."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                rows.append((row[0].strip(), float(row[1]), float(row[2])))
            except ValueError:
                continue
    return rows


def summarize(rows):
    """Mean/median/max of inference, post-processing, and total seconds."""
    inference = [r[1] for r in rows]
    post = [r[2] for r in rows]
    total = [i + p for i, p in zip(inference, post)]
    return {
        "n": len(rows),
        "inference_mean": mean(inference), "inference_median": median(inference),
        "inference_max": max(inference),
        "post_mean": mean(post), "post_median": median(post),
        "post_max": max(post),
        "total_mean": mean(total), "total_median": median(total),
        "total_max": max(total),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, nargs="+",
                    help="model output dir(s), each containing a timings.csv")
    ap.add_argument("--output-dir",
                    help="if given, write the summary table there as timing_summary.csv")
    args = ap.parse_args()

    results = []
    for input_dir in args.input_dir:
        path = os.path.join(input_dir, "timings.csv")
        if not os.path.isfile(path):
            print(f"WARNING: no timings.csv in {input_dir} "
                  "(run the model with --output-dir first) — skipped.")
            continue
        rows = read_timings(path)
        if not rows:
            print(f"WARNING: {path} has no timing rows — skipped.")
            continue
        results.append((input_dir, summarize(rows)))

    if not results:
        sys.exit("No timings.csv found in any input dir.")

    header = ["model", "n",
              "inference_mean_s", "inference_median_s", "inference_max_s",
              "post_mean_s", "post_median_s", "post_max_s",
              "total_mean_s", "total_median_s", "total_max_s"]
    table = [[name, s["n"],
              f"{s['inference_mean']:.3f}", f"{s['inference_median']:.3f}",
              f"{s['inference_max']:.3f}",
              f"{s['post_mean']:.3f}", f"{s['post_median']:.3f}",
              f"{s['post_max']:.3f}",
              f"{s['total_mean']:.3f}", f"{s['total_median']:.3f}",
              f"{s['total_max']:.3f}"]
             for name, s in results]

    print("\n=== Timing summary (seconds per image) ===")
    name_w = max(len("model"), max(len(r[0]) for r in table))
    print(f"{'model':<{name_w}}{'n':>5}"
          f"{'inf mean':>10}{'inf med':>10}{'inf max':>10}"
          f"{'post mean':>11}{'post med':>10}{'post max':>10}"
          f"{'tot mean':>10}{'tot med':>10}{'tot max':>10}")
    print("-" * (name_w + 96))
    for r in table:
        print(f"{r[0]:<{name_w}}{r[1]:>5}"
              f"{r[2]:>10}{r[3]:>10}{r[4]:>10}"
              f"{r[5]:>11}{r[6]:>10}{r[7]:>10}"
              f"{r[8]:>10}{r[9]:>10}{r[10]:>10}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "timing_summary.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"\nWrote summary -> {out}")


if __name__ == "__main__":
    main()
