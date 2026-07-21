"""Compare model-predicted leaf *masks* against ground-truth leaf masks.

Unlike evaluate_leaf_count.py (which only compares counts), this scores mask
*overlap quality* using the CVPPP Leaf Segmentation Challenge (LSC) metrics:

    FBD     Foreground-Background Dice -- Dice between the union of all
            predicted masks and the GT foreground blob. Measures "did it find
            the plant", independent of how well individual leaves are split.
    SBD     Symmetric Best Dice -- for each GT leaf, best Dice to any
            predicted mask (and vice versa), averaged, keeping the worse of
            the two directions. The standard CVPPP leaderboard number.
    AP@t    Precision / Recall / F1 from a greedy one-to-one match between
            predicted and GT leaf instances at IoU >= t (default 0.5).

CAVEAT on AP: the `--masks` PNGs written by the run scripts are hard binary
masks with no per-instance confidence score, so there is no precision-recall
curve to integrate -- this is a single-threshold operating point (P/R/F1 @
IoU>=t), not a confidence-ranked COCO-style mAP@[.5:.95]. Matching is greedy
by descending IoU rather than an optimal (Hungarian) assignment, to avoid a
scipy dependency; at typical leaf counts per image this differs from the
optimal assignment vanishingly rarely.

Needs numpy + Pillow (requirements-common.txt) -- unlike the stdlib-only
count evaluator, mask math needs array support.

Ground truth: `data/cvppp/mask/A?/plantNNN_fg.png` (binary foreground) and
`data/cvppp/per_leaf_mask/A?/plantNNN_label.png` (0=background, 1..N=leaf id).
Either may be omitted to skip the metrics that need it (FBD needs only the
foreground masks; SBD and AP need only the label maps).

Predictions: a `--masks` output dir, i.e. a tree containing `<image>_masks/
mask_*.png` subfolders written by any 2D run script's `--masks` flag.

Rows are matched by canonical plant id, stripping known suffixes
(`_masks`, `_rgb`, `_fg`, `_label`) and the extension, so
`plant001_rgb_masks/`, `plant001_fg.png` and `plant001_label.png` all line up
as `plant001`. Evaluate one CVPPP subset (A1/A2/A3/A4) at a time -- like the
count evaluator, A1/A2/A3 reuse `plantNNN` ids for different plants.

Examples:
    python eval/evaluate_leaf_segmentation.py \\
        --pred-masks 2d_foundation/01_sam2/outputs/A1 \\
        --gt-fg data/cvppp/mask/A1 \\
        --gt-label data/cvppp/per_leaf_mask/A1

    # FBD only (no per-instance ground truth needed)
    python eval/evaluate_leaf_segmentation.py \\
        --pred-masks 2d_foundation/01_sam2/outputs/A1 --gt-fg data/cvppp/mask/A1

    # full per-image table + stricter IoU
    python eval/evaluate_leaf_segmentation.py \\
        --pred-masks 2d_foundation/01_sam2/outputs/A1 \\
        --gt-label data/cvppp/per_leaf_mask/A1 --iou-thresh 0.75 \\
        --per-image --out results.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from shared.leafviz import masks_from_label_map

SUFFIXES = ("_masks", "_leaves", "_rgb", "_fg", "_label")


def _canonical_id(name):
    """Strip extension and any of `_masks`/`_leaves`/`_rgb`/`_fg`/`_label`, so
    predictions (`plant001_rgb_masks`) and ground truth (`plant001_fg.png`,
    `plant001_label.png`) line up on `plant001`."""
    stem = os.path.splitext(os.path.basename(str(name).strip()))[0]
    for suf in SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def load_pred_masks(pred_root):
    """{plant_id: [bool (H,W) mask, ...]} from every `<stem>_masks/` folder."""
    dirs = [d for d in glob.glob(os.path.join(pred_root, "**", "*_masks"), recursive=True)
            if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"No '*_masks' folders found under {pred_root} "
                 "(run a model with --masks first).")
    out = {}
    for d in sorted(dirs):
        key = _canonical_id(os.path.basename(d))
        files = sorted(glob.glob(os.path.join(d, "mask_*.png")))
        out[key] = [np.asarray(Image.open(f).convert("L")) > 127 for f in files]
    return out


def load_gt_fg(gt_dir):
    """{plant_id: bool (H,W) foreground mask} from `plantNNN_fg.png` files."""
    out = {}
    for f in glob.glob(os.path.join(gt_dir, "*.png")):
        out[_canonical_id(f)] = np.asarray(Image.open(f).convert("L")) > 0
    return out


def load_gt_label(gt_dir):
    """{plant_id: [bool (H,W) mask, ...]} from `plantNNN_label.png` label maps."""
    out = {}
    for f in glob.glob(os.path.join(gt_dir, "*.png")):
        arr = np.asarray(Image.open(f))
        out[_canonical_id(f)] = masks_from_label_map(arr)
    return out


def _resize_like(mask, shape):
    """Nearest-neighbour resize a bool mask to `shape`, if it doesn't already match."""
    if mask.shape == shape:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def fbd_for_image(pred_masks, gt_fg):
    shape = gt_fg.shape
    pred_union = np.zeros(shape, dtype=bool)
    for m in pred_masks:
        pred_union |= _resize_like(m, shape)
    return dice(pred_union, gt_fg)


def sbd_for_image(pred_masks, gt_masks):
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    shape = gt_masks[0].shape
    pred_masks = [_resize_like(m, shape) for m in pred_masks]
    dice_mat = np.array([[dice(g, p) for p in pred_masks] for g in gt_masks])
    forward = dice_mat.max(axis=1).mean()   # each GT leaf -> best predicted leaf
    backward = dice_mat.max(axis=0).mean()  # each predicted leaf -> best GT leaf
    return min(forward, backward)


def match_instances(pred_masks, gt_masks, iou_thresh):
    """Greedy one-to-one match by descending IoU. Returns (tp, fp, fn)."""
    if not gt_masks and not pred_masks:
        return 0, 0, 0
    if not gt_masks:
        return 0, len(pred_masks), 0
    if not pred_masks:
        return 0, 0, len(gt_masks)
    shape = gt_masks[0].shape
    pred_masks = [_resize_like(m, shape) for m in pred_masks]
    iou_mat = np.array([[iou(p, g) for g in gt_masks] for p in pred_masks])
    pairs = sorted(
        ((iou_mat[i, j], i, j) for i in range(len(pred_masks)) for j in range(len(gt_masks))),
        reverse=True,
    )
    matched_pred, matched_gt, tp = set(), set(), 0
    for score, i, j in pairs:
        if score < iou_thresh:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        tp += 1
    return tp, len(pred_masks) - tp, len(gt_masks) - tp


def evaluate(pred, gt_fg, gt_label, iou_thresh):
    keys = sorted(pred)
    if gt_fg is not None:
        keys = [k for k in keys if k in gt_fg]
    if gt_label is not None:
        keys = [k for k in keys if k in gt_label]
    rows = []
    for k in keys:
        row = {"image": k}
        if gt_fg is not None:
            row["fbd"] = fbd_for_image(pred[k], gt_fg[k])
        if gt_label is not None:
            row["sbd"] = sbd_for_image(pred[k], gt_label[k])
            row["tp"], row["fp"], row["fn"] = match_instances(pred[k], gt_label[k], iou_thresh)
        rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    stats = {"n": n}
    if "fbd" in rows[0]:
        stats["FBD"] = sum(r["fbd"] for r in rows) / n
    if "sbd" in rows[0]:
        stats["SBD"] = sum(r["sbd"] for r in rows) / n
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats.update({"precision": precision, "recall": recall, "f1": f1,
                      "tp": tp, "fp": fp, "fn": fn})
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-masks", required=True,
                    help="root dir containing <image>_masks/mask_*.png folders "
                         "(from a run script's --masks flag)")
    ap.add_argument("--gt-fg", help="dir of plantNNN_fg.png ground-truth foreground masks")
    ap.add_argument("--gt-label", help="dir of plantNNN_label.png ground-truth instance label maps")
    ap.add_argument("--iou-thresh", type=float, default=0.5,
                    help="IoU threshold for the AP@t instance match (default 0.5)")
    ap.add_argument("--per-image", action="store_true", help="print the full per-image table")
    ap.add_argument("--out", help="write per-image results to this CSV")
    args = ap.parse_args()

    if not args.gt_fg and not args.gt_label:
        sys.exit("Pass at least one of --gt-fg (for FBD) or --gt-label (for SBD/AP).")

    pred = load_pred_masks(args.pred_masks)
    gt_fg = load_gt_fg(args.gt_fg) if args.gt_fg else None
    gt_label = load_gt_label(args.gt_label) if args.gt_label else None
    rows = evaluate(pred, gt_fg, gt_label, args.iou_thresh)

    if args.per_image and rows:
        cols = ["image"] + [c for c in ("fbd", "sbd", "tp", "fp", "fn") if c in rows[0]]
        print("".join(f"{c:<10}" if c != "image" else f"{c:<24}" for c in cols))
        print("-" * (24 + 10 * (len(cols) - 1)))
        for r in rows:
            print("".join(
                f"{r[c]:<24}" if c == "image" else
                f"{r[c]:<10.3f}" if isinstance(r[c], float) else f"{r[c]:<10}"
                for c in cols))
        print()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        cols = ["image"] + [c for c in ("fbd", "sbd", "tp", "fp", "fn") if rows and c in rows[0]]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote per-image results -> {args.out}")

    n_pred, n_fg, n_label = len(pred), len(gt_fg or {}), len(gt_label or {})
    print(f"predictions : {args.pred_masks}  ({n_pred} images)")
    if gt_fg is not None:
        print(f"gt fg       : {args.gt_fg}  ({n_fg} images)")
    if gt_label is not None:
        print(f"gt label    : {args.gt_label}  ({n_label} images)")
    print(f"matched     : {len(rows)} images")

    stats = summarize(rows)
    if not stats:
        sys.exit("\nNo overlapping images between predictions and ground truth "
                 "— check that image basenames match (see docstring on id matching).")

    print("\n=== CVPPP leaf-segmentation metrics ===")
    if "FBD" in stats:
        print(f"  FBD              : {stats['FBD']:.3f}")
    if "SBD" in stats:
        print(f"  SBD              : {stats['SBD']:.3f}")
        print(f"  AP@{args.iou_thresh:.2f} precision  : {stats['precision']:.3f}")
        print(f"  AP@{args.iou_thresh:.2f} recall     : {stats['recall']:.3f}")
        print(f"  AP@{args.iou_thresh:.2f} F1         : {stats['f1']:.3f}"
              f"   (tp={stats['tp']} fp={stats['fp']} fn={stats['fn']})")


if __name__ == "__main__":
    main()
