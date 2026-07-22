"""Draw the YOLO-seg polygon labels onto a copy of every image, for eyeballing them.

Reads the images/ + labels/ tree that masks_to_yolo.py produced and writes, into
a SEPARATE folder, a copy of each image with its leaf polygons drawn on top. The
originals under --data are never modified. Browse the result with any image
viewer (feh, eog) or by clicking the files in VS Code:

    <out>/train/A1_plant006_rgb.png   <out>/val/...

Example:
    python draw_labels.py                # yolo_dataset/ -> label_overlays/
    feh label_overlays/train
"""
import argparse
import colorsys
import glob
import os

import cv2
import numpy as np


def make_palette(n=20):
    """n visually distinct BGR colors (cv2 uses BGR order)."""
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.75, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def draw_one(image, label_path, colors, alpha):
    """Return a new image with every polygon in label_path drawn on (semi-transparent)."""
    out = image.copy()
    fill = image.copy()
    h, w = image.shape[:2]
    for i, line in enumerate(open(label_path).read().splitlines()):
        if not line.strip():
            continue
        coords = np.array(line.split()[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w                       # denormalize x (labels are 0..1)
        coords[:, 1] *= h                       # denormalize y
        pts = coords.astype(np.int32)
        color = colors[i % len(colors)]
        cv2.fillPoly(fill, [pts], color)        # solid fill on one copy
        cv2.polylines(out, [pts], True, color, 2)   # crisp outline on the other
    cv2.addWeighted(fill, alpha, out, 1 - alpha, 0, out)   # blend the fill in
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="yolo_dataset",
                    help="dataset root holding images/ and labels/ (read-only here)")
    ap.add_argument("--out", default="label_overlays",
                    help="separate folder to write the drawn-on copies into")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="polygon fill opacity (0 = outline only, 1 = solid)")
    args = ap.parse_args()

    colors = make_palette()
    total = 0
    for split in ("train", "val"):
        img_dir = os.path.join(args.data, "images", split)
        lbl_dir = os.path.join(args.data, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)

        n = 0
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            name = os.path.basename(img_path)
            label_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            image = cv2.imread(img_path)            # a fresh copy; original file untouched
            if os.path.exists(label_path):
                image = draw_one(image, label_path, colors, args.alpha)
            cv2.imwrite(os.path.join(out_dir, name), image)
            n += 1
        print(f"  {split}: wrote {n} overlays -> {out_dir}")
        total += n

    if total == 0:
        raise SystemExit(f"No images under {args.data}/images/ -- run masks_to_yolo.py first.")
    print(f"Done: {total} overlays in {args.out}/  (browse with: feh {args.out}/train)")


if __name__ == "__main__":
    main()
