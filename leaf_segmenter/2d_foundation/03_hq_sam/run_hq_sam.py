"""Zero-shot leaf segmentation with HQ-SAM (Segment Anything in High Quality).

HQ-SAM refines SAM's masks for sharper boundaries — useful for thin/serrated
leaf edges. Same automatic-mask + green-filter recipe as folder 01, but the
higher-quality decoder. `vit_tiny` = "Light HQ-SAM" (fast, fits 6 GB easily).

Checkpoints auto-download from the Hugging Face hub (`lkeab/hq-sam`).

Examples:
    python run_hq_sam.py --image ../../data/cvppp/images/A1/plant001_rgb.png --model-type vit_tiny
    python run_hq_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b --output outputs/
    python run_hq_sam.py --image branch.jpg --crops   # cut out each leaf
    python run_hq_sam.py --image branch.jpg --masks   # binary masks, for SBD/AP eval
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
                            save_counts_csv, save_image, write_crops, write_masks)

HF_REPO = "lkeab/hq-sam"
CKPT_FILES = {
    "vit_tiny": "sam_hq_vit_tiny.pth",  # Light HQ-SAM
    "vit_b": "sam_hq_vit_b.pth",
    "vit_l": "sam_hq_vit_l.pth",
    "vit_h": "sam_hq_vit_h.pth",
}


def ensure_checkpoint(model_type, explicit):
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(HF_REPO, CKPT_FILES[model_type])
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Could not auto-download {CKPT_FILES[model_type]} from {HF_REPO} ({e}).\n"
            "Download it manually (see README) and pass --checkpoint /path/to.pth"
        )


def build_generator(model_type, ckpt, device, points_per_side):
    from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    return SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)


def filter_leaf_masks(image, anns, min_green, min_area, max_area):
    h, w = image.shape[:2]
    total = h * w
    kept = []
    for a in anns:
        seg = a["segmentation"].astype(bool)
        area = seg.sum()
        if area < min_area * total or area > max_area * total:
            continue
        if min_green > 0 and green_fraction(image, seg) < min_green:
            continue
        kept.append(seg)
    return kept


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
    out_path = os.path.join(out_dir, f"{stem}_hqsam.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(anns)} raw -> {len(masks)} kept "
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--input-dir")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--model-type", choices=list(CKPT_FILES), default="vit_tiny")
    ap.add_argument("--checkpoint", help="path to sam_hq_*.pth (else auto-download)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--no-green-filter", action="store_true")
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
    ap.add_argument("--min-green", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.0005)
    ap.add_argument("--max-area", type=float, default=0.25)
    args = ap.parse_args()

    inputs = gather_inputs(args)
    if not inputs:
        sys.exit("No input images found.")
    os.makedirs(args.output, exist_ok=True)

    ckpt = ensure_checkpoint(args.model_type, args.checkpoint)
    print(f"Loading HQ-SAM ({args.model_type}) on {args.device} ...")
    gen = build_generator(args.model_type, ckpt, args.device, args.points_per_side)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p), process(p, gen, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
