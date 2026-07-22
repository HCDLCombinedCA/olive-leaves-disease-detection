"""YOLOv8-seg / YOLO11-seg instance segmentation (Ultralytics).

Fast, easy, strong. COCO-pretrained weights (`yolo11n-seg.pt` etc.) auto-download.
COCO has no "leaf" class — pass `--weights` pointing at a model you fine-tuned
with train_yolo_seg.py for real leaf results.

Usage:
    python run_yolo_seg.py --input-dir ../../data/cvppp/images/A1
    python run_yolo_seg.py --input-dir ../../data/cvppp/images/A1 --weights best.pt --output-dir out

With --output-dir, each input image gets its own folder holding the colour
overlay and one binary PNG per instance, plus a shared counts.csv. `run()`
always returns, per input image, the list of transparent-background cutouts.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import (crop_leaves, list_images, load_image, save_outputs,
                            save_timings_csv)

# ---- constants (previously CLI flags) --------------------------------------
DEVICE = 0 if torch.cuda.is_available() else "cpu"   # 0 = first GPU
CONF = 0.25            # min detection confidence
IMGSZ = 640            # inference image size


def extract_masks(res, image):
    """Return a YOLO result's instance masks at the original image resolution."""
    if res.masks is None:
        return []
    h, w = image.shape[:2]
    masks = []
    for m in res.masks.data.cpu().numpy():   # (N, h', w') float
        if m.shape != (h, w):
            from PIL import Image
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))) > 127
        else:
            m = m > 0.5
        masks.append(m)
    return masks


def run(input_dir, weights="yolo11n-seg.pt", output_dir=None):
    """Segment instances in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background cutouts. With
    `output_dir`, also writes each image's colour overlay + per-instance binary
    masks (one folder per image) and a shared counts.csv, via
    shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    from ultralytics import YOLO

    print(f"Loading YOLO-seg ({weights}) on device {DEVICE} ...")
    model = YOLO(weights)

    all_crops, all_masks, timings = [], [], []
    for path in paths:
        image = load_image(path)

        # inference
        t0 = time.perf_counter()
        res = model.predict(path, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        t1 = time.perf_counter()

        # post processing
        masks = extract_masks(res, image)
        t2 = time.perf_counter()

        # saving of data
        image_name = os.path.basename(path)
        timings.append((image_name, t1 - t0, t2 - t1))
        all_crops.append({"image_name": image_name,
                          "cropped_images": crop_leaves(image, masks)})
        all_masks.append({"image_name": image_name, "image": image, "masks": masks})
        print(f"{image_name}: {len(masks)} instances (conf>={CONF})")

    if output_dir:
        save_outputs(output_dir, all_masks, "yoloseg")
        save_timings_csv(os.path.join(output_dir, "timings.csv"), timings)
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--weights", default="yolo11n-seg.pt",
                    help="COCO model auto-downloads; or pass your fine-tuned best.pt")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
