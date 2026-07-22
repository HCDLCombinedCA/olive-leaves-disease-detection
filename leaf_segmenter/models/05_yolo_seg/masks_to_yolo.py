"""Convert the CVPPP per-leaf label maps into the YOLO-seg polygon format.

YOLO-seg can't read the label-map PNGs; it needs, per image, a .txt with one
line per leaf: `0 x1 y1 x2 y2 ...` (class 0 = leaf, coordinates normalized to
[0, 1]). This script traces each leaf's outline into such a polygon and lays the
files out the way Ultralytics expects, then writes leaf.yaml pointing at them.

    <out>/images/train/A1_plant006_rgb.png   <out>/labels/train/A1_plant006_rgb.txt
    <out>/images/val/...                      <out>/labels/val/...

Filenames are prefixed with the subset (A1_, A2_, ...) because the CVPPP subsets
reuse plantNNN names -- without the prefix they'd overwrite each other.

Example:
    python masks_to_yolo.py                     # ../../data/cvppp -> yolo_dataset/
    python train_yolo_seg.py --data leaf.yaml --model yolo11n-seg.pt --epochs 100 --batch 4
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image

# which source split becomes which YOLO split
SPLITS = {"fine_tuning": "train", "testing": "val"}


def mask_to_polygon(mask, w, h, epsilon):
    """Trace one leaf's boolean mask into a normalized polygon string, or None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)          # biggest blob if the leaf fragmented
    if epsilon > 0:
        c = cv2.approxPolyDP(c, epsilon, True)  # fewer points, same shape
    c = c.reshape(-1, 2)
    if len(c) < 3:
        return None                             # need >= 3 points for a polygon
    return "0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in c)


def label_map_to_lines(label_path, epsilon):
    """All leaf polygons in one label map, as a list of YOLO-seg text lines."""
    label_map = np.array(Image.open(label_path))
    h, w = label_map.shape
    lines = []
    for leaf_id in np.unique(label_map):
        if leaf_id == 0:                        # 0 = background
            continue
        mask = (label_map == leaf_id).astype(np.uint8)
        line = mask_to_polygon(mask, w, h, epsilon)
        if line:
            lines.append(line)
    return lines


def convert_split(data_root, src, dst, out, epsilon):
    """Convert one source split (e.g. fine_tuning -> train). Returns #images written."""
    label_paths = sorted(glob.glob(
        os.path.join(data_root, src, "per_leaf_mask", "*", "*_label.png")))
    if not label_paths:
        print(f"  no label maps under {data_root}/{src} -- skipped")
        return 0

    img_out = os.path.join(out, "images", dst)
    lbl_out = os.path.join(out, "labels", dst)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    for label_path in label_paths:
        subset = os.path.basename(os.path.dirname(label_path))       # A1, A2, ...
        stem = os.path.basename(label_path).replace("_label.png", "_rgb")
        image_path = os.path.join(data_root, src, "images", subset, stem + ".png")
        if not os.path.exists(image_path):
            continue

        lines = label_map_to_lines(label_path, epsilon)
        name = f"{subset}_{stem}"                                    # A1_plant006_rgb
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        shutil.copy(image_path, os.path.join(img_out, name + ".png"))
        count += 1

    print(f"  {src} -> {dst}: {count} images")
    return count


def write_yaml(yaml_path, out):
    """Write the Ultralytics dataset yaml pointing at the converted data."""
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: leaf\n")
    print(f"Wrote {yaml_path} (path -> {os.path.abspath(out)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="../../data/cvppp",
                    help="folder holding the fine_tuning/ and testing/ splits")
    ap.add_argument("--out", default="yolo_dataset",
                    help="where to write the YOLO images/ and labels/ tree")
    ap.add_argument("--yaml", default="leaf.yaml", help="dataset yaml to write")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="polygon simplification in pixels (0 = keep every point)")
    args = ap.parse_args()

    total = 0
    for src, dst in SPLITS.items():
        total += convert_split(args.data_root, src, dst, args.out, args.epsilon)
    if total == 0:
        raise SystemExit(f"No images converted -- check --data-root ({args.data_root}).")

    write_yaml(args.yaml, args.out)
    print(f"Done: {total} images. Now: python train_yolo_seg.py --data {args.yaml} "
          "--model yolo11n-seg.pt --epochs 100 --batch 4")


if __name__ == "__main__":
    main()
