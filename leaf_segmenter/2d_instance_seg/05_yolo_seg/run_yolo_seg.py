"""YOLOv8-seg / YOLO11-seg instance segmentation (Ultralytics).

Fast, easy, strong. In orchard tests YOLOv8-seg beat Mask R-CNN on trunks and
branches, and YOLO11-seg improves on YOLOv8 for occluded objects.

COCO-pretrained weights (`yolo11n-seg.pt` etc.) auto-download. Like Mask R-CNN,
COCO has no "leaf" class — use `--weights` to point at a model you fine-tuned
with train_yolo_seg.py for real leaf results.

Examples:
    python run_yolo_seg.py --image ../../data/samples/synthetic_leaf.png --output outputs/
    python run_yolo_seg.py --input-dir ../../data/samples --weights yolo11s-seg.pt
    python run_yolo_seg.py --image branch.jpg --weights runs/segment/train/weights/best.pt
    python run_yolo_seg.py --image branch.jpg --crops   # cut out each instance
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.leafviz import (load_image, overlay_masks, save_counts_csv,
                            save_image, write_crops)


def gather_inputs(args):
    if args.image:
        return [args.image]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.input_dir, e))
        files += glob.glob(os.path.join(args.input_dir, e.upper()))
    return sorted(set(files))


def process(path, model, args, out_dir):
    image = load_image(path)
    t0 = time.time()
    res = model.predict(path, conf=args.conf, imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
    dt = time.time() - t0

    masks = []
    if res.masks is not None:
        # resize mask data back to original resolution
        h, w = image.shape[:2]
        for m in res.masks.data.cpu().numpy():  # (N, h', w') float
            mm = np.asarray(m)
            if mm.shape != (h, w):
                from PIL import Image
                mm = np.asarray(Image.fromarray((mm * 255).astype(np.uint8)).resize((w, h)))
                mm = mm > 127
            else:
                mm = mm > 0.5
            masks.append(mm)

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_yoloseg.png")
    save_image(overlay_masks(image, masks), out_path)
    msg = (f"{os.path.basename(path)}: {len(masks)} instances (conf>={args.conf}) "
           f"in {dt:.1f}s  ->  {out_path}")

    if args.crops:
        crop_dir = os.path.join(out_dir, f"{stem}_leaves")
        n = write_crops(image, masks, crop_dir, stem)
        msg += f"  ->  {n} instance crops + full overlay in {crop_dir}/"

    print(msg)
    return len(masks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--input-dir")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--weights", default="yolo11n-seg.pt",
                    help="COCO model auto-downloads; or pass your fine-tuned best.pt")
    ap.add_argument("--device", default=0, help="'0' for first GPU, or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
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

    from ultralytics import YOLO

    print(f"Loading YOLO-seg ({args.weights}) ...")
    model = YOLO(args.weights)
    counts = []
    for p in inputs:
        counts.append((os.path.basename(p), process(p, model, args, args.output)))
    if args.count_csv:
        save_counts_csv(args.count_csv, counts)
        print(f"Wrote {len(counts)} counts -> {args.count_csv}")


if __name__ == "__main__":
    main()
