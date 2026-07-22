"""Mask2Former instance segmentation (Hugging Face transformers).

Transformer-based, state-of-the-art on the CVPPP leaf benchmark. Runs the
COCO-instance-pretrained model by default (weights auto-download). As with the
other COCO models there is no "leaf" class out of the box — fine-tune on a leaf
dataset for real results, or point --model at your fine-tuned checkpoint.

Usage:
    python run_mask2former.py --input-dir ../../data/cvppp/images/A1
    python run_mask2former.py --input-dir ../../data/cvppp/images/A1 --model facebook/mask2former-swin-base-coco-instance --output-dir out

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
THRESHOLD = 0.5        # min confidence to keep an instance
DEFAULT_MODEL = "facebook/mask2former-swin-small-coco-instance"  # swin-small fits 6 GB


def predict(processor, model, image):
    """Run Mask2Former on one image and return its raw outputs."""
    from PIL import Image

    inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        return model(**inputs)


def extract_masks(processor, outputs, image):
    """Turn raw Mask2Former outputs into per-instance boolean masks."""
    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.shape[:2]], threshold=THRESHOLD
    )[0]
    seg = result["segmentation"].cpu().numpy()   # (H, W) int, -1 = no object
    return [seg == s["id"] for s in result["segments_info"]]


def run(input_dir, model_name=DEFAULT_MODEL, output_dir=None):
    """Segment instances in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background cutouts. With
    `output_dir`, also writes each image's colour overlay + per-instance binary
    masks (one folder per image) and a shared counts.csv, via
    shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    print(f"Loading Mask2Former ({model_name}) on {DEVICE} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).eval().to(DEVICE)

    all_crops, all_masks, timings = [], [], []
    for path in paths:
        image = load_image(path)

        # inference
        t0 = time.perf_counter()
        outputs = predict(processor, model, image)
        t1 = time.perf_counter()

        # post processing
        masks = extract_masks(processor, outputs, image)
        t2 = time.perf_counter()

        # saving of data
        image_name = os.path.basename(path)
        timings.append((image_name, t1 - t0, t2 - t1))
        all_crops.append({"image_name": image_name,
                          "cropped_images": crop_leaves(image, masks)})
        all_masks.append({"image_name": image_name, "image": image, "masks": masks})
        print(f"{image_name}: {len(masks)} instances (thr={THRESHOLD})")

    if output_dir:
        save_outputs(output_dir, all_masks, "mask2former")
        save_timings_csv(os.path.join(output_dir, "timings.csv"), timings)
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="swin-small fits 6 GB; -base/-large need more VRAM")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.model, args.output_dir)


if __name__ == "__main__":
    main()
