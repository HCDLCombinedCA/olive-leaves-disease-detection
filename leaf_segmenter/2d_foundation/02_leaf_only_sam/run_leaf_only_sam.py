"""Leaf Only SAM — zero-shot leaf segmentation by post-processing SAM's masks.

Reimplementation of the pipeline from Williams et al., "Leaf Only SAM: A Segment
Anything Pipeline for Zero-Shot Automated Leaf Segmentation" (2023). It runs the
original Segment Anything (SAM v1) automatic mask generator, then applies a
series of filters to keep only leaf-shaped, green, non-background masks:

  1. drop masks that are too large / too small (background & noise)
  2. drop masks that hug the image border (pot / background)
  3. keep masks whose pixels are mostly green (foliage color)
  4. drop masks that are heavily contained in another kept mask (duplicates)

No training or annotation required. SAM v1 weights auto-download on first run.

Examples:
    python run_leaf_only_sam.py --image ../../data/samples/synthetic_leaf.png --output outputs/
    python run_leaf_only_sam.py --input-dir ../../data/samples --model-type vit_b
    python run_leaf_only_sam.py --image branch.jpg --crops   # cut out each leaf
"""
import argparse
import glob
import os
import sys
import time
import urllib.request

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.leafviz import (green_fraction, load_image, overlay_masks,
                            save_counts_csv, save_image, write_crops)

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


def touches_border(mask: np.ndarray, frac: float) -> bool:
    """True if the mask covers > `frac` of any image edge (background/pot)."""
    h, w = mask.shape
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return edges.mean() > frac


def dedup_by_containment(masks, iou_contain=0.8):
    """Drop a mask if >= iou_contain of it lies inside an already-kept (larger) one."""
    order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)
    kept = []
    for i in order:
        m = masks[i]
        a = m.sum()
        redundant = False
        for k in kept:
            inter = np.logical_and(m, k).sum()
            if a > 0 and inter / a > iou_contain:
                redundant = True
                break
        if not redundant:
            kept.append(m)
    return kept


def leaf_only_filter(image, anns, args):
    h, w = image.shape[:2]
    total = h * w
    candidates = []
    for a in anns:
        seg = a["segmentation"].astype(bool)
        area = seg.sum()
        if area < args.min_area * total or area > args.max_area * total:
            continue
        if touches_border(seg, args.border_frac):
            continue
        if green_fraction(image, seg) < args.min_green:
            continue
        candidates.append(seg)
    return dedup_by_containment(candidates, args.contain)


def build_generator(model_type, ckpt, device, points_per_side):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    return SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)


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
    # filter thresholds (paper-style post-processing)
    ap.add_argument("--min-green", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.0005)
    ap.add_argument("--max-area", type=float, default=0.2)
    ap.add_argument("--border-frac", type=float, default=0.15,
                    help="drop mask if it covers > this fraction of an image edge")
    ap.add_argument("--contain", type=float, default=0.8,
                    help="dedup: drop mask if this fraction of it sits inside a bigger one")
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
