"""Zero-shot leaf segmentation with Meta's Segment Anything Model 2 (SAM 2).

Automatic mode: SAM2AutomaticMaskGenerator segments *everything* in the image,
then we (optionally) keep only the green, mid-sized masks as candidate leaves.
No training or annotation required. Weights auto-download from Hugging Face on
first run.

Examples:
    python run_sam2.py --image ../../data/samples/synthetic_leaf.png --output outputs/
    python run_sam2.py --input-dir ../../data/samples --model-size tiny --output outputs/
    python run_sam2.py --image branch.jpg --no-green-filter   # keep every mask
    python run_sam2.py --image branch.jpg --crops             # cut out each leaf
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.leafviz import (green_fraction, load_image, overlay_masks,
                            save_counts_csv, save_image, write_crops)

MODEL_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def build_mask_generator(model_id: str, device: str, points_per_side: int):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Newer sam2 exposes .from_pretrained on the generator; fall back to build_sam2_hf.
    try:
        return SAM2AutomaticMaskGenerator.from_pretrained(
            model_id, device=device, points_per_side=points_per_side
        )
    except (AttributeError, TypeError):
        from sam2.build_sam import build_sam2_hf

        model = build_sam2_hf(model_id, device=device)
        return SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side)


def filter_leaf_masks(image, anns, min_green, min_area_frac, max_area_frac):
    h, w = image.shape[:2]
    total = h * w
    keep = []
    for a in anns:
        seg = a["segmentation"].astype(bool)
        area = seg.sum()
        if area < min_area_frac * total or area > max_area_frac * total:
            continue
        if min_green > 0 and green_fraction(image, seg) < min_green:
            continue
        keep.append(seg)
    return keep


def process(path, generator, args, out_dir):
    image = load_image(path)
    t0 = time.time()
    with torch.inference_mode():
        anns = generator.generate(image)
    dt = time.time() - t0

    if args.no_green_filter:
        masks = [a["segmentation"].astype(bool) for a in anns]
    else:
        masks = filter_leaf_masks(image, anns, args.min_green, args.min_area, args.max_area)

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_sam2.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(anns)} raw masks -> {len(masks)} kept "
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
    src.add_argument("--image", help="single image path")
    src.add_argument("--input-dir", help="folder of images")
    ap.add_argument("--output", default="outputs", help="output folder")
    ap.add_argument("--model-size", choices=list(MODEL_IDS), default="small",
                    help="tiny/small fit a 6 GB GPU comfortably; large may OOM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--points-per-side", type=int, default=32,
                    help="lower (e.g. 16) = faster / less memory, coarser masks")
    ap.add_argument("--no-green-filter", action="store_true",
                    help="keep ALL of SAM's masks instead of foliage-only")
    ap.add_argument("--crops", action="store_true",
                    help="also write per-leaf transparent-PNG cutouts plus a "
                         "full all-masks overlay into outputs/<image>_leaves/")
    ap.add_argument("--count-csv",
                    help="write predicted leaf counts (image,n_leaves) here, "
                         "for eval/evaluate_leaf_count.py")
    ap.add_argument("--min-green", type=float, default=0.5,
                    help="min fraction of green pixels for a mask to count as a leaf")
    ap.add_argument("--min-area", type=float, default=0.0005,
                    help="drop masks smaller than this fraction of the image")
    ap.add_argument("--max-area", type=float, default=0.25,
                    help="drop masks larger than this fraction (usually background)")
    args = ap.parse_args()

    inputs = gather_inputs(args)
    if not inputs:
        sys.exit("No input images found.")
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading SAM 2 ({MODEL_IDS[args.model_size]}) on {args.device} ...")
    gen = build_mask_generator(MODEL_IDS[args.model_size], args.device, args.points_per_side)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p), process(p, gen, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
