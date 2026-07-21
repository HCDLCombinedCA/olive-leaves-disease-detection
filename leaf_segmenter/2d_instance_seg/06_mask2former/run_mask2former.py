"""Mask2Former instance segmentation (Hugging Face transformers).

Transformer-based, state-of-the-art on the CVPPP leaf benchmark. Runs the
COCO-instance-pretrained model by default (weights auto-download). As with the
other COCO models there is no "leaf" class out of the box — fine-tune on a leaf
dataset for real results (see README), or swap in your fine-tuned checkpoint via
--model.

Examples:
    python run_mask2former.py --image ../../data/cvppp/images/A1/plant001_rgb.png --output outputs/
    python run_mask2former.py --input-dir ../../data/cvppp/images/A1 --model facebook/mask2former-swin-base-coco-instance
    python run_mask2former.py --image branch.jpg --crops   # cut out each instance
    python run_mask2former.py --image branch.jpg --masks   # binary masks, for SBD/AP eval
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
from shared.leafviz import (load_image, overlay_masks, save_counts_csv,
                            save_image, write_crops, write_masks)


def gather_inputs(args):
    if args.image:
        return [args.image]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.input_dir, e))
        files += glob.glob(os.path.join(args.input_dir, e.upper()))
    return sorted(set(files))


def process(path, processor, model, device, args, out_dir):
    from PIL import Image

    image = load_image(path)
    pil = Image.fromarray(image)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.inference_mode():
        outputs = model(**inputs)
    dt = time.time() - t0

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.shape[:2]], threshold=args.threshold
    )[0]
    seg = result["segmentation"].cpu().numpy()  # (H, W) int, -1 = no object
    masks = [seg == s["id"] for s in result["segments_info"]]

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_mask2former.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(masks)} instances "
           f"(thr={args.threshold}) in {dt:.1f}s  ->  {out_path}")

    if args.crops:
        crop_dir = os.path.join(out_dir, f"{stem}_leaves")
        n = write_crops(image, masks, crop_dir, stem)
        msg += f"  ->  {n} instance crops + full overlay in {crop_dir}/"

    if args.masks:
        mask_dir = os.path.join(out_dir, f"{stem}_masks")
        n_m = write_masks(masks, mask_dir, stem)
        msg += f"  ->  {n_m} binary masks in {mask_dir}/"

    print(msg)
    return len(masks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--input-dir")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--model", default="facebook/mask2former-swin-small-coco-instance",
                    help="swin-small fits 6 GB; -base/-large need more VRAM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min confidence to keep an instance")
    ap.add_argument("--crops", action="store_true",
                    help="also write per-instance transparent-PNG cutouts plus a "
                         "full all-masks overlay into outputs/<image>_leaves/")
    ap.add_argument("--count-csv",
                    help="write predicted instance counts (image,n_leaves) here, "
                         "for eval/evaluate_leaf_count.py")
    ap.add_argument("--masks", action="store_true",
                    help="also write one full-size binary PNG per instance mask "
                         "(white=leaf, black=background) into "
                         "outputs/<image>_masks/, for SBD/AP-style evaluation")
    args = ap.parse_args()

    inputs = gather_inputs(args)
    if not inputs:
        sys.exit("No input images found.")
    os.makedirs(args.output, exist_ok=True)

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    print(f"Loading Mask2Former ({args.model}) on {args.device} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).eval().to(args.device)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p),
                       process(p, processor, model, args.device, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
