"""Zero-shot leaf segmentation with HQ-SAM (Segment Anything in High Quality).

HQ-SAM refines SAM's masks for sharper boundaries — useful for thin/serrated
leaf edges. Same automatic-mask + green-filter recipe as folder 01, but the
higher-quality decoder. `vit_tiny` = "Light HQ-SAM" (fast, fits 6 GB easily).
Checkpoints auto-download from the Hugging Face hub (`lkeab/hq-sam`).

Usage:
    python run_hq_sam.py --input-dir ../../data/cvppp/images/A1
    python run_hq_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b --output-dir out

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

HF_REPO = "lkeab/hq-sam"
CKPT_FILES = {
    "vit_tiny": "sam_hq_vit_tiny.pth",  # Light HQ-SAM
    "vit_b": "sam_hq_vit_b.pth",
    "vit_l": "sam_hq_vit_l.pth",
    "vit_h": "sam_hq_vit_h.pth",
}


def ensure_checkpoint(model_type):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(HF_REPO, CKPT_FILES[model_type])
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Could not auto-download {CKPT_FILES[model_type]} from {HF_REPO} ({e}).\n"
            "Download it manually (see README) and place it in the HF cache."
        )


def build_generator(model_type, ckpt):
    from segment_anything_hq import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=ckpt).to(DEVICE)
    return SamAutomaticMaskGenerator(sam, points_per_side=POINTS_PER_SIDE)


def filter_leaf_masks(image, anns):
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


def run(input_dir, model_type="vit_tiny", output_dir=None):
    """Segment leaves in every image in `input_dir`.

    Returns `all_crops`: one dict per image, {"image_name", "cropped_images"},
    where cropped_images is the list of transparent-background leaf cutouts. With
    `output_dir`, also writes each image's colour overlay + per-leaf binary masks
    (one folder per image) and a shared counts.csv, via shared.helper.save_outputs.
    """
    paths = list_images(input_dir)

    print(f"Loading HQ-SAM ({model_type}) on {DEVICE} ...")
    gen = build_generator(model_type, ensure_checkpoint(model_type))

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
        print(f"{image_name}: {len(anns)} raw -> {len(masks)} kept")

    if output_dir:
        save_outputs(output_dir, all_masks, "hqsam")
    return all_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--model-type", choices=list(CKPT_FILES), default="vit_tiny")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    run(args.input_dir, args.model_type, args.output_dir)


if __name__ == "__main__":
    main()
