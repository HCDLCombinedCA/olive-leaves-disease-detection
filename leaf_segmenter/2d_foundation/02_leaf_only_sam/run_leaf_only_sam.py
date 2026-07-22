"""Leaf Only SAM — zero-shot leaf segmentation by post-processing SAM's masks.

Runs the original Segment Anything (SAM v1) automatic mask generator with the
paper's settings (Williams et al., 2023), then applies the paper's four filters
to keep only leaves: green colour, whole-plant removal, leaf shape, and composite
removal. SAM v1 weights auto-download on first run.

Usage:
    python run_leaf_only_sam.py --input-dir ../../data/cvppp/images/A1
    python run_leaf_only_sam.py --input-dir imgs --model-type vit_l --output-dir out

With --output-dir, each input image gets its own folder holding the colour
overlay and one binary PNG per leaf, plus a shared counts.csv. `run()` always
returns, per input image, the list of transparent-background leaf cutouts.
"""
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import crop_leaves, list_images, load_image, save_outputs

# ---- constants (previously CLI flags) --------------------------------------
CKPT_DIR = "checkpoints"
POINTS_PER_SIDE = 32
MIN_SHAPE = 0.1        # checkshape: min contour-area / enclosing-circle area
SUBSET_THRESH = 0.9    # remove_toobig: overlap fraction to count as contained
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPTS = {
    "vit_b": ("sam_vit_b_01ec64.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_h": ("sam_vit_h_4b8939.pth",
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
}


def ensure_checkpoint(model_type):
    fname, url = CKPTS[model_type]
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, path)
    return path


# ---- the paper's four leaf filters -----------------------------------------
def _select(masks, keep):
    return [m for m, k in zip(masks, keep) if k]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def checkcolour(hsv, masks):
    """Keep masks whose mean HSV colour is green (grow-light fallback widens hue)."""
    means = np.zeros((len(masks), 3))
    for i, m in enumerate(masks):
        if m.any():
            means[i] = hsv[m].mean(axis=0)
    h, s = means[:, 0], means[:, 1]
    idx = (h < 75) & (h > 35) & (s > 35)
    if idx.sum() == 0:                       # grow-light fallback: widen hue band
        idx = (h < 100) & (h > 35) & (s > 35)
    return idx


def checkfullplant(masks):
    """Drop any mask that is ~the union of all masks (the whole-plant blob)."""
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    return np.array([iou(m, union) < 0.9 for m in masks])


def shape_ratio(mask):
    """Compactness: largest-contour area / its min-enclosing-circle area."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=len)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    carea = np.pi * radius ** 2
    return cv2.contourArea(cnt) / carea if carea > 0 else 0.0


def remove_toobig(masks):
    """Drop composite masks that are just their own smaller sub-masks stacked."""
    keep = [True] * len(masks)
    for i, big in enumerate(masks):
        big_area = int(big.sum())
        if big_area == 0:
            keep[i] = False
            continue
        covered = np.zeros_like(big)
        found = False
        for j, other in enumerate(masks):
            if i == j or not keep[j]:
                continue
            oa = int(other.sum())
            if oa and np.logical_and(other, big).sum() / oa > SUBSET_THRESH:
                covered |= other
                found = True
        if found and np.logical_and(big, covered).sum() / big_area > 0.9:
            keep[i] = False
    return _select(masks, keep)


def leaf_only_filter(image, anns):
    masks = [a["segmentation"].astype(bool) for a in anns]
    if not masks:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   # paper feeds RGB through BGR->HSV
    masks = _select(masks, checkcolour(hsv, masks))          # 1. green colour
    if len(masks) > 2:                                       # 2. drop whole-plant mask
        masks = _select(masks, checkfullplant(masks))
    masks = [m for m in masks if shape_ratio(m) > MIN_SHAPE]  # 3. leaf shape
    return remove_toobig(masks)                              # 4. drop composites


# ---- run SAM over a folder --------------------------------------------------
def build_generator(model_type, ckpt):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=ckpt).to(DEVICE)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )


def run(input_dir, model_type="vit_b", output_dir=None):
    """Segment leaves in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background leaf cutouts. With
    `output_dir`, also writes each image's colour overlay + per-leaf binary masks
    (one folder per image) and a shared counts.csv, via shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    gen = build_generator(model_type, ensure_checkpoint(model_type))
    print(f"Loaded SAM ({model_type}) on {DEVICE}")

    all_crops, all_masks = [], []
    for path in paths:
        image = load_image(path)

        with torch.inference_mode():
            anns = gen.generate(image)
        masks = leaf_only_filter(image, anns)

        image_name = os.path.basename(path)
        all_crops.append({"image_name": image_name,
                          "cropped_images": crop_leaves(image, masks)})
        all_masks.append({"image_name": image_name, "image": image, "masks": masks})
        print(f"{image_name}: {len(anns)} raw -> {len(masks)} leaves")

    if output_dir:
        save_outputs(output_dir, all_masks, "leafonlysam")
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--model-type", choices=list(CKPTS), default="vit_b")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.model_type, args.output_dir)


if __name__ == "__main__":
    main()
