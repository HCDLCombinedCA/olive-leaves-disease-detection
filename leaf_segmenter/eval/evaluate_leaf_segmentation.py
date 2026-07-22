"""Compare model-predicted leaf *masks* against ground-truth leaf masks (CVPPP LSC).

Unlike evaluate_leaf_count.py (which only compares counts), this scores mask
*overlap quality*:

    FBD     Foreground-Background Dice -- Dice between the union of all predicted
            masks and the GT foreground (the union of all GT leaves). "Did it find
            the plant?", independent of how well individual leaves are split.
    SBD     Symmetric Best Dice -- for each GT leaf, best Dice to any predicted
            mask (and vice versa), keeping the worse direction. The standard CVPPP
            leaf-splitting number.
    AP@t    Precision / Recall / F1 from a greedy one-to-one match between predicted
            and GT leaf instances at IoU >= t (t = IOU_THRESH, default 0.5).

Predictions are read from `<input-dir>/<stem>/mask_*.png` -- the per-image folders
any run script writes with --output-dir. Ground truth is read from the per-leaf
label maps under --gt-dir (`plantNNN_label.png`, 0=background, 1..N=leaf id); the
FBD foreground is derived from those maps (label > 0), so no separate foreground
file is needed. Rows are matched by canonical plant id (the `_masks`/`_rgb`/`_fg`/
`_label` suffixes and the extension are stripped, so `plant001_rgb/` and
`plant001_label.png` line up on `plant001`). Evaluate one CVPPP subset
(A1/A2/A3/A4) at a time -- they reuse `plantNNN` ids for different plants.

With --output-dir, also writes the full per-image table there. Needs numpy +
Pillow (requirements-common.txt).

CAVEAT on AP: the mask PNGs are hard binary masks with no per-instance confidence,
so this is a single-threshold operating point (P/R/F1 @ IoU>=t), not a
confidence-ranked COCO-style mAP. Matching is greedy by descending IoU (no scipy
dependency); at typical leaf counts this differs from optimal assignment rarely.

Examples:
    python eval/evaluate_leaf_segmentation.py \\
        --input-dir 2d_foundation/01_sam2/output/small/A1 \\
        --gt-dir data/cvppp/per_leaf_mask/A1
    python eval/evaluate_leaf_segmentation.py \\
        --input-dir out/A1 --gt-dir data/cvppp/per_leaf_mask/A1 --output-dir eval_out/A1
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
from shared.helper import masks_from_label_map

IOU_THRESH = 0.5    # IoU threshold for the AP@t instance match
SUFFIXES = ("_masks", "_leaves", "_rgb", "_fg", "_label")


def _canonical_id(name):
    """Strip extension and any known suffix so predictions (`plant001_rgb`) and
    ground truth (`plant001_label.png`) line up on `plant001`."""
    stem = os.path.splitext(os.path.basename(str(name).strip()))[0]
    for suf in SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def load_pred_masks(input_dir):
    """{plant_id: [bool (H,W) mask, ...]} from each `<input-dir>/<stem>/` folder."""
    dirs = sorted(d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d))
    if not dirs:
        sys.exit(f"No per-image folders under {input_dir} "
                 "(run a model with --output-dir first).")
    out = {}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "mask_*.png")))
        out[_canonical_id(os.path.basename(d))] = [
            np.asarray(Image.open(f).convert("L")) > 127 for f in files]
    return out


def load_gt(gt_dir):
    """{plant_id: [bool leaf mask, ...]} from `plantNNN_label.png` instance maps."""
    out = {}
    for f in sorted(glob.glob(os.path.join(gt_dir, "*_label.png"))):
        out[_canonical_id(f)] = masks_from_label_map(np.asarray(Image.open(f)))
    if not out:
        sys.exit(f"No *_label.png ground-truth maps under {gt_dir}")
    return out


def _resize_like(mask, shape):
    """Nearest-neighbour resize a bool mask to `shape`, if it doesn't already match."""
    if mask.shape == shape:
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


def dice(a, b):
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * np.logical_and(a, b).sum() / denom


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def fbd_for_image(pred_masks, gt_masks):
    shape = gt_masks[0].shape
    pred_union = np.zeros(shape, dtype=bool)
    for m in pred_masks:
        pred_union |= _resize_like(m, shape)
    gt_union = np.zeros(shape, dtype=bool)
    for m in gt_masks:
        gt_union |= m
    return dice(pred_union, gt_union)


def sbd_for_image(pred_masks, gt_masks):
    if not gt_masks and not pred_masks:
        return 1.0
    if not gt_masks or not pred_masks:
        return 0.0
    shape = gt_masks[0].shape
    pred_masks = [_resize_like(m, shape) for m in pred_masks]
    dm = np.array([[dice(g, p) for p in pred_masks] for g in gt_masks])
    return min(dm.max(axis=1).mean(), dm.max(axis=0).mean())


def match_instances(pred_masks, gt_masks):
    """Greedy one-to-one match by descending IoU. Returns (tp, fp, fn)."""
    if not gt_masks and not pred_masks:
        return 0, 0, 0
    if not gt_masks:
        return 0, len(pred_masks), 0
    if not pred_masks:
        return 0, 0, len(gt_masks)
    shape = gt_masks[0].shape
    pred_masks = [_resize_like(m, shape) for m in pred_masks]
    pairs = sorted(((iou(p, g), i, j)
                    for i, p in enumerate(pred_masks) for j, g in enumerate(gt_masks)),
                   reverse=True)
    matched_pred, matched_gt, tp = set(), set(), 0
    for score, i, j in pairs:
        if score < IOU_THRESH:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        tp += 1
    return tp, len(pred_masks) - tp, len(gt_masks) - tp


def evaluate(pred, gt):
    """Per-image FBD / SBD / (tp, fp, fn) over the shared plant ids."""
    rows = []
    for k in sorted(pred):
        if k not in gt:
            continue
        tp, fp, fn = match_instances(pred[k], gt[k])
        rows.append({"image": k,
                     "fbd": fbd_for_image(pred[k], gt[k]),
                     "sbd": sbd_for_image(pred[k], gt[k]),
                     "tp": tp, "fp": fp, "fn": fn})
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return None
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": n,
            "FBD": sum(r["fbd"] for r in rows) / n,
            "SBD": sum(r["sbd"] for r in rows) / n,
            "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="model output dir with <image>/mask_*.png per-image folders")
    ap.add_argument("--gt-dir", required=True,
                    help="folder of CVPPP plantNNN_label.png per-leaf instance maps")
    ap.add_argument("--output-dir",
                    help="if given, write the full per-image FBD/SBD/tp/fp/fn table here")
    args = ap.parse_args()

    pred = load_pred_masks(args.input_dir)
    gt = load_gt(args.gt_dir)
    rows = evaluate(pred, gt)

    print(f"predictions : {args.input_dir}  ({len(pred)} images)")
    print(f"ground truth: {args.gt_dir}  ({len(gt)} images)")
    print(f"matched     : {len(rows)} images")

    stats = summarize(rows)
    if not stats:
        sys.exit("\nNo overlapping images between predictions and ground truth "
                 "— check that image basenames match.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "leaf_segmentation_per_image.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "fbd", "sbd", "tp", "fp", "fn"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n{'image':<24}{'fbd':>8}{'sbd':>8}{'tp':>5}{'fp':>5}{'fn':>5}")
        print("-" * 55)
        for r in rows:
            print(f"{r['image']:<24}{r['fbd']:>8.3f}{r['sbd']:>8.3f}"
                  f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}")
        print(f"\nWrote per-image table -> {out}")

    print("\n=== CVPPP leaf-segmentation metrics ===")
    print(f"  FBD              : {stats['FBD']:.3f}")
    print(f"  SBD              : {stats['SBD']:.3f}")
    print(f"  AP@{IOU_THRESH:.2f} precision  : {stats['precision']:.3f}")
    print(f"  AP@{IOU_THRESH:.2f} recall     : {stats['recall']:.3f}")
    print(f"  AP@{IOU_THRESH:.2f} F1         : {stats['f1']:.3f}"
          f"   (tp={stats['tp']} fp={stats['fp']} fn={stats['fn']})")


if __name__ == "__main__":
    main()
