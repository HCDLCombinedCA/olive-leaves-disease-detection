"""Split the CVPPP dataset into a fine-tuning set (random 10%) and a testing set
(the remaining 90%), keeping each image together with its masks and count.

Run it from inside the cvppp folder:
    cd leaf_segmenter/data/cvppp
    python ../split_dataset.py

For every subset (A1-A4) it copies each plant's three files -- the RGB image, the
foreground mask, and the per-leaf mask -- plus the per-subset count CSV (filtered
to just that split's images), into:
    fine_tuning/{images,mask,per_leaf_mask}/<subset>/...   (random 10%)
    testing/{images,mask,per_leaf_mask}/<subset>/...       (the other 90%)
Files are copied, not moved, so the originals are left untouched. The split is
reproducible via SEED.
"""
import csv
import glob
import os
import random
import shutil

FINE_TUNE_FRACTION = 0.10
SEED = 0

# each plant has one file per category folder, distinguished by this suffix
CATEGORIES = {"images": "_rgb.png", "mask": "_fg.png", "per_leaf_mask": "_label.png"}


def _plant_id(name):
    """plantNNN id from an image reference ('plant001_rgb.png' -> 'plant001')."""
    base = os.path.basename(str(name).strip())
    return base[:-len("_rgb.png")] if base.endswith("_rgb.png") else os.path.splitext(base)[0]


def split_subset(subset):
    ids = sorted(f[: -len("_rgb.png")]
                 for f in os.listdir(os.path.join("images", subset))
                 if f.endswith("_rgb.png"))
    random.shuffle(ids)
    n = round(len(ids) * FINE_TUNE_FRACTION)
    gt_csvs = glob.glob(os.path.join("images", subset, "*.csv"))
    for split, split_ids in (("fine_tuning", ids[:n]), ("testing", ids[n:])):
        id_set = set(split_ids)
        # copy the image + its two masks
        for category, suffix in CATEGORIES.items():
            dst = os.path.join(split, category, subset)
            os.makedirs(dst, exist_ok=True)
            for pid in split_ids:
                shutil.copy2(os.path.join(category, subset, pid + suffix),
                             os.path.join(dst, pid + suffix))
        # copy the count CSV, keeping only this split's rows
        for src in gt_csvs:
            with open(src, newline="") as f:
                rows = [r for r in csv.reader(f) if r and _plant_id(r[0]) in id_set]
            with open(os.path.join(split, "images", subset, os.path.basename(src)),
                      "w", newline="") as f:
                csv.writer(f).writerows(rows)
    return n, len(ids) - n


def main():
    if not os.path.isdir("images"):
        raise SystemExit("Run this from inside the cvppp folder "
                         "(the one containing images/, mask/, per_leaf_mask/).")
    random.seed(SEED)
    subsets = sorted(d for d in os.listdir("images")
                     if os.path.isdir(os.path.join("images", d)))
    for subset in subsets:
        n_ft, n_te = split_subset(subset)
        print(f"{subset}: {n_ft} fine-tuning, {n_te} testing")


if __name__ == "__main__":
    main()
