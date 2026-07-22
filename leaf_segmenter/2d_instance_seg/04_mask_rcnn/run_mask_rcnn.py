"""Mask R-CNN instance segmentation (torchvision, COCO-pretrained).

Mask R-CNN is the long-time standard for the CVPPP leaf-segmentation challenge.
This script runs the torchvision implementation with COCO weights (the easiest
Mask R-CNN to install). COCO has no "leaf" class, so out of the box this finds
generic objects; its real purpose is as a runnable baseline and the pretrained
backbone you FINE-TUNE on a leaf dataset (pass the .pth via --weights).

Usage:
    python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1
    python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --weights finetuned.pth --output-dir out

With --output-dir, each input image gets its own folder holding the colour
overlay and one binary PNG per instance, plus a shared counts.csv. `run()`
always returns, per input image, the list of transparent-background cutouts.
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import (crop_leaves, list_images, load_image, save_outputs,
                            save_timings_csv)

# ---- constants (previously CLI flags) --------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2        # only used with --weights (incl. background)
SCORE_THRESH = 0.5     # min detection confidence to keep an instance
MASK_THRESH = 0.5      # threshold on the soft mask to binarise it


def build_model(weights_path):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if weights_path:
        # your fine-tuned model: build with matching #classes, then load state dict
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    return model.eval().to(DEVICE)


def run(input_dir, weights=None, output_dir=None):
    """Segment instances in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background cutouts. With
    `output_dir`, also writes each image's colour overlay + per-instance binary
    masks (one folder per image) and a shared counts.csv, via
    shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    print(f"Loading Mask R-CNN on {DEVICE} "
          f"({'fine-tuned ' + weights if weights else 'COCO-pretrained'}) ...")
    model = build_model(weights)

    all_crops, all_masks, timings = [], [], []
    for path in paths:
        image = load_image(path)

        # inference
        t0 = time.perf_counter()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(DEVICE)
        with torch.inference_mode():
            pred = model([tensor])[0]
        t1 = time.perf_counter()

        # post processing
        keep = pred["scores"].cpu().numpy() >= SCORE_THRESH
        masks = list(pred["masks"].cpu().numpy()[keep, 0] > MASK_THRESH)  # (N, H, W) bool
        t2 = time.perf_counter()

        # saving of data
        image_name = os.path.basename(path)
        timings.append((image_name, t1 - t0, t2 - t1))
        all_crops.append({"image_name": image_name,
                          "cropped_images": crop_leaves(image, masks)})
        all_masks.append({"image_name": image_name, "image": image, "masks": masks})
        print(f"{image_name}: {len(masks)} instances (score>={SCORE_THRESH})")

    if output_dir:
        save_outputs(output_dir, all_masks, "maskrcnn")
        save_timings_csv(os.path.join(output_dir, "timings.csv"), timings)
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--weights", help="fine-tuned state_dict (.pth); omit for COCO weights")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
