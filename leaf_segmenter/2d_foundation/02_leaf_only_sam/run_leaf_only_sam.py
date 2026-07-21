"""Leaf Only SAM — zero-shot leaf segmentation by post-processing SAM's masks.

Reimplementation of the pipeline from Williams et al., "Leaf Only SAM: A Segment
Anything Pipeline for Zero-Shot Automated Leaf Segmentation" (2023). It runs the
original Segment Anything (SAM v1) automatic mask generator with the paper's
settings, then applies the paper's post-processing to keep only leaves:

  1. keep masks whose average HSV colour is green -- checkcolour, with a
     grow-light fallback that widens the hue band if nothing passes
  2. if the plant was split into > 2 masks, drop any mask that is basically the
     whole plant / union of all masks -- checkfullplant
  3. keep only compact, leaf-shaped masks -- contour area vs. its minimum
     enclosing circle -- checkshape
  4. drop big composite masks that are just their smaller sub-masks stacked
     together, keeping the individual leaves -- istoobig / remove_toobig

This mirrors the reference notebook (github.com/Dom3442/leafonlysam). It needs
OpenCV (cv2) as well as numpy: the SAM generator's min_mask_region_area and the
contour / enclosing-circle shape filter both use it.

Deviations from the notebook, on purpose: images are loaded via Pillow and kept
at full resolution (the notebook downsamples 0.5x for GPU memory) so the binary
masks line up pixel-for-pixel with the CVPPP ground truth for evaluation; manage
memory with --model-type / --points-per-side instead.

No training or annotation required. SAM v1 weights auto-download on first run.

Examples:
    python run_leaf_only_sam.py --image ../../data/cvppp/images/A1/plant001_rgb.png --output outputs/
    python run_leaf_only_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b
    python run_leaf_only_sam.py --image branch.jpg --crops   # cut out each leaf
    python run_leaf_only_sam.py --image branch.jpg --masks   # binary masks, for SBD/AP eval
"""
import argparse
import glob
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.leafviz import (load_image, overlay_masks, save_counts_csv,
                            save_image, write_crops, write_masks)

# ---- SAM v1 checkpoints (auto-downloaded on first use) ---------------------
CKPTS = {
    "vit_b": ("sam_vit_b_01ec64.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_h": ("sam_vit_h_4b8939.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
}


def ensure_checkpoint(model_type: str, ckpt_dir: str) -> str:
    fname, url = CKPTS[model_type]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, path)
    return path


# ---- the paper's four leaf filters (run in order by leaf_only_filter) ------
def _select(masks, keep):
    """Keep the masks where the parallel boolean array `keep` is True."""
    return [m for m, k in zip(masks, keep) if k]


def iou(a, b):
    """Intersection-over-union of two boolean masks (0..1)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def checkcolour(hsv, masks):
    """Paper's colour filter: keep masks whose *mean* HSV colour is green.

    For each mask, average the HSV pixels inside it and keep it if the mean hue
    is in the green band (OpenCV hue is 0-179; 35-75 ~ yellow-green..green-cyan)
    and it is saturated enough not to be washed-out background. If nothing
    passes, widen the hue ceiling to 100 -- the paper's "grow lights on" fallback
    for the colour cast of purple/pink horticultural LEDs. Returns a bool array.
    """
    means = np.zeros((len(masks), 3), dtype=float)
    for i, m in enumerate(masks):
        if m.any():
            means[i] = hsv[m].mean(axis=0)
    h, s = means[:, 0], means[:, 1]
    idx_green = (h < 75) & (h > 35) & (s > 35)
    if idx_green.sum() == 0:                      # grow-light fallback: widen hue band
        idx_green = (h < 100) & (h > 35) & (s > 35)
    return idx_green


def checkfullplant(masks):
    """Paper's checkfullplant: drop any mask that is ~the union of all masks.

    Once the plant is split into several leaf masks, SAM often also returns one
    mask covering the whole plant. Any mask whose IoU with the union of every
    mask is >= 0.9 is that whole-plant blob and is dropped. Returns a bool array.
    """
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    return np.array([iou(m, union) < 0.9 for m in masks])


def shape_ratio(mask):
    """Paper's checkshape compactness: contour area / min-enclosing-circle area.

    Traces the mask's largest contour and compares its area to the area of its
    minimum enclosing circle (cv2.minEnclosingCircle, as in the paper). ~1.0 for
    a filled disk, near 0 for a thin/straggly sliver; the pipeline keeps masks
    above --min-shape (paper default 0.1).
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=len)                      # paper: contour with the most points
    area = cv2.contourArea(cnt)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    carea = np.pi * radius ** 2
    return float(area / carea) if carea > 0 else 0.0


def remove_toobig(masks, subset_thresh=0.9):
    """Paper's istoobig/remove_toobig: drop composite masks, keep the leaves.

    A mask is a redundant "too big" composite if the other masks that are almost
    entirely inside it (>= `subset_thresh` of each such mask lies within it)
    together cover more than 90% of it -- i.e. it is just its own sub-leaves
    stacked together. Such masks are removed so the individual leaves survive.
    (This is the opposite of a naive containment de-dup, which would keep the big
    blob; it is what stops a whole plant collapsing into one mask.)
    """
    keep = [True] * len(masks)
    for i, big in enumerate(masks):                # `big`: is it just smaller masks stacked?
        big_area = int(big.sum())
        if big_area == 0:
            keep[i] = False
            continue
        covered = np.zeros_like(big)               # union of masks that sit inside `big`
        found_sub = False
        for j, other in enumerate(masks):
            if i == j or not keep[j]:
                continue
            oa = int(other.sum())
            if oa and np.logical_and(other, big).sum() / oa > subset_thresh:
                covered |= other                   # `other` is ~entirely inside `big`
                found_sub = True
        # if those inner masks tile > 90% of `big`, it is a composite -> drop it
        if found_sub and np.logical_and(big, covered).sum() / big_area > 0.9:
            keep[i] = False
    return _select(masks, keep)


def leaf_only_filter(image, anns, args):
    masks = [a["segmentation"].astype(bool) for a in anns]
    if not masks:
        return []
    # The paper feeds its RGB array through cv2's BGR->HSV; reproduce that exactly
    # so the hardcoded hue thresholds in checkcolour match its calibration.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = _select(masks, checkcolour(hsv, masks))          # 1. green colour
    if len(masks) > 2:                                       # 2. drop whole-plant mask
        masks = _select(masks, checkfullplant(masks))
    masks = [m for m in masks if shape_ratio(m) > args.min_shape]  # 3. leaf shape
    return remove_toobig(masks, args.subset_thresh)          # 4. drop composites


# ---- run SAM and write the results -----------------------------------------
def build_generator(model_type, ckpt, device, points_per_side):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    # Generator settings match the paper's notebook (crop layer + small-region
    # cleanup improve results on the smallest leaves). min_mask_region_area needs
    # cv2.
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )


def process(path, generator, args, out_dir):
    image = load_image(path)
    t0 = time.time()
    with torch.inference_mode():
        anns = generator.generate(image)
    dt = time.time() - t0
    masks = leaf_only_filter(image, anns, args)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_leafonlysam.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(anns)} raw -> {len(masks)} leaves "
           f"in {dt:.1f}s  ->  {out_path}")

    if args.crops:
        crop_dir = os.path.join(out_dir, f"{stem}_leaves")
        all_masks = [a["segmentation"].astype(bool) for a in anns]
        n = write_crops(image, masks, crop_dir, stem, all_masks=all_masks)
        msg += f"  ->  {n} leaf crops + full overlay in {crop_dir}/"

    if args.masks:
        mask_dir = os.path.join(out_dir, f"{stem}_masks")
        n_m = write_masks(masks, mask_dir, stem)
        msg += f"  ->  {n_m} binary masks in {mask_dir}/"

    print(msg)
    return len(masks)


def gather_inputs(args):
    if args.image:
        return [args.image]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.input_dir, e))
        files += glob.glob(os.path.join(args.input_dir, e.upper()))
    return sorted(set(files))


# ---- command-line interface ------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--input-dir")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--model-type", choices=list(CKPTS), default="vit_b",
                    help="vit_b fits 6 GB; vit_l/h need more VRAM")
    ap.add_argument("--checkpoint", help="path to a SAM .pth (else auto-download)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--crops", action="store_true",
                    help="also write per-leaf transparent-PNG cutouts plus a "
                         "full all-masks overlay into outputs/<image>_leaves/")
    ap.add_argument("--count-csv",
                    help="write predicted leaf counts (image,n_leaves) here, "
                         "for eval/evaluate_leaf_count.py")
    ap.add_argument("--masks", action="store_true",
                    help="also write one full-size binary PNG per leaf mask "
                         "(white=leaf, black=background) into "
                         "outputs/<image>_masks/, for SBD/AP-style evaluation")
    # post-processing filter thresholds (paper defaults)
    ap.add_argument("--min-shape", type=float, default=0.1,
                    help="paper checkshape: min contour-area / enclosing-circle "
                         "area to keep a mask (drops thin/straggly non-leaf blobs)")
    ap.add_argument("--subset-thresh", type=float, default=0.9,
                    help="paper istoobig/remove_toobig: overlap fraction for one "
                         "mask to count as contained in another (composite removal)")
    args = ap.parse_args()

    inputs = gather_inputs(args)
    if not inputs:
        sys.exit("No input images found.")
    os.makedirs(args.output, exist_ok=True)

    ckpt = args.checkpoint or ensure_checkpoint(args.model_type, args.ckpt_dir)
    print(f"Loading SAM ({args.model_type}) on {args.device} ...")
    gen = build_generator(args.model_type, ckpt, args.device, args.points_per_side)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p), process(p, gen, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
