"""Zero-shot leaf segmentation with Meta's Segment Anything Model 2 (SAM 2).

Automatic mode: SAM2AutomaticMaskGenerator segments *everything* in the image,
then we keep only the green, mid-sized masks as candidate leaves. No training or
annotation required. Weights auto-download from Hugging Face on first run.

Usage:
    python run_sam2.py --input-dir ../../data/cvppp/images/A1
    python run_sam2.py --input-dir ../../data/cvppp/images/A1 --model-size tiny --output-dir out

With --output-dir, each input image gets its own folder holding the colour
overlay and one binary PNG per leaf, plus a shared counts.csv. `run()` always
returns, per input image, the list of transparent-background leaf cutouts.
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import (crop_leaves, green_fraction, list_images,
                            load_image, save_outputs)

# ---- constants (previously CLI flags) --------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POINTS_PER_SIDE = 32
MIN_GREEN = 0.5        # min fraction of green pixels for a mask to count as a leaf
MIN_AREA = 0.0005      # drop masks smaller than this fraction of the image
MAX_AREA = 0.25        # drop masks larger than this fraction (usually background)

MODEL_IDS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def build_mask_generator(model_id):
    """Load pretrained SAM 2 weights and wrap them in an automatic mask generator."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Newer sam2 exposes .from_pretrained on the generator; fall back to build_sam2_hf.
    try:
        return SAM2AutomaticMaskGenerator.from_pretrained(
            model_id, device=DEVICE, points_per_side=POINTS_PER_SIDE
        )
    except (AttributeError, TypeError):
        from sam2.build_sam import build_sam2_hf

        model = build_sam2_hf(model_id, device=DEVICE)
        return SAM2AutomaticMaskGenerator(model, points_per_side=POINTS_PER_SIDE)


def filter_leaf_masks(image, anns):
    """Keep only mid-sized, mostly-green masks -- SAM 2 is class-agnostic, so this
    is how we pick the leaves out of its "segment everything" output."""
    total = image.shape[0] * image.shape[1]
    kept = []
    for a in anns:
        seg = a["segmentation"].astype(bool)
        area = seg.sum()
        if area < MIN_AREA * total or area > MAX_AREA * total:
            continue
        if MIN_GREEN > 0 and green_fraction(image, seg) < MIN_GREEN:
            continue
        kept.append(seg)
    return kept


def run(input_dir, model_size="small", output_dir=None):
    """Segment leaves in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background leaf cutouts. With
    `output_dir`, also writes each image's colour overlay + per-leaf binary masks
    (one folder per image) and a shared counts.csv, via shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    print(f"Loading SAM 2 ({MODEL_IDS[model_size]}) on {DEVICE} ...")
    gen = build_mask_generator(MODEL_IDS[model_size])

    all_crops, all_masks = [], []
    for path in paths:
        image = load_image(path)
        with torch.inference_mode():
            anns = gen.generate(image)
        masks = filter_leaf_masks(image, anns)

        image_name = os.path.basename(path)
        all_crops.append({"image_name": image_name,
                          "cropped_images": crop_leaves(image, masks)})
        all_masks.append({"image_name": image_name, "image": image, "masks": masks})
        print(f"{image_name}: {len(anns)} raw masks -> {len(masks)} kept")

    if output_dir:
        save_outputs(output_dir, all_masks, "sam2")
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--model-size", choices=list(MODEL_IDS), default="small")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.model_size, args.output_dir)


if __name__ == "__main__":
    main()
