"""Mask R-CNN instance segmentation (torchvision, COCO-pretrained).

Mask R-CNN is the long-time standard for the CVPPP leaf-segmentation challenge.
This script runs the torchvision implementation with COCO weights — the easiest
Mask R-CNN to install (no Detectron2 build headaches).

IMPORTANT: COCO has no "leaf" class, so out of the box this finds generic
objects (and usually nothing on a plain branch photo). Its real purpose here is:
  (a) a runnable Mask R-CNN baseline / sanity check, and
  (b) the pretrained backbone you FINE-TUNE on a leaf dataset (CVPPP, Poplar-leaf).
See README for the fine-tuning path and the Detectron2 / CSIRO alternative.

Examples:
    python run_mask_rcnn.py --image ../../data/samples/synthetic_leaf.png --output outputs/
    python run_mask_rcnn.py --input-dir ../../data/samples --score-thresh 0.3
    python run_mask_rcnn.py --image branch.jpg --weights path/to/finetuned.pth --num-classes 2
    python run_mask_rcnn.py --image branch.jpg --crops   # cut out each instance
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
                            save_image, write_crops)


def build_model(device, weights_path, num_classes):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if weights_path:
        # your fine-tuned model: build with matching #classes, then load state dict
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    return model.eval().to(device)


def process(path, model, device, args, out_dir):
    image = load_image(path)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)
    t0 = time.time()
    with torch.inference_mode():
        pred = model([tensor])[0]
    dt = time.time() - t0

    scores = pred["scores"].cpu().numpy()
    keep = scores >= args.score_thresh
    masks = pred["masks"].cpu().numpy()[keep, 0] > args.mask_thresh  # (N, H, W) bool
    masks = [m for m in masks]

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_maskrcnn.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(masks)} instances "
           f"(score>={args.score_thresh}) in {dt:.1f}s  ->  {out_path}")

    if args.crops:
        crop_dir = os.path.join(out_dir, f"{stem}_leaves")
        n = write_crops(image, masks, crop_dir, stem)
        msg += f"  ->  {n} instance crops + full overlay in {crop_dir}/"

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
    ap.add_argument("--weights", help="fine-tuned state_dict (.pth); omit for COCO weights")
    ap.add_argument("--num-classes", type=int, default=2,
                    help="only used with --weights (incl. background)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--crops", action="store_true",
                    help="also write per-instance transparent-PNG cutouts plus a "
                         "full all-masks overlay into outputs/<image>_leaves/")
    ap.add_argument("--count-csv",
                    help="write predicted instance counts (image,n_leaves) here, "
                         "for eval/evaluate_leaf_count.py")
    args = ap.parse_args()

    inputs = gather_inputs(args)
    if not inputs:
        sys.exit("No input images found.")
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading Mask R-CNN on {args.device} "
          f"({'fine-tuned '+args.weights if args.weights else 'COCO-pretrained'}) ...")
    model = build_model(args.device, args.weights, args.num_classes)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p), process(p, model, args.device, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
