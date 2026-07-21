# 02 — Leaf Only SAM

**What:** A zero-shot pipeline that runs the original **SAM (v1)** automatic mask
generator with the paper's settings, then applies the paper's four post-processing
filters to isolate leaves: (1) HSV green-colour, (2) whole-plant-mask removal,
(3) contour compactness/shape, (4) composite-mask removal. No training/annotation
needed.

**Reference numbers (from the paper, potato leaves):** recall 63.2 / precision
60.3 zero-shot, vs. 78.7 / 74.7 for a fine-tuned Mask R-CNN — i.e. a fine-tuned
model still wins, but this needs zero labels.

**Status:** ✅ Runs out of the box. `run_leaf_only_sam.py` mirrors the reference
notebook (`Dom3442/leafonlysam`) — SAM generator settings and all four filters
(`checkcolour`, `checkfullplant`, `checkshape`, `istoobig`/`remove_toobig`). SAM
v1 weights auto-download. Needs OpenCV (in `requirements.txt`). The one deliberate
deviation: images are kept at full resolution (the notebook downsamples 0.5×) so
the output masks align with the CVPPP ground truth for evaluation.

**Reference paper:** [Leaf Only SAM: A Segment Anything Pipeline for Zero-Shot Automated Leaf Segmentation (Williams et al., 2023)](https://arxiv.org/abs/2305.09418)

## Install

```bash
cd 2d_foundation/02_leaf_only_sam
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python run_leaf_only_sam.py --image ../../data/cvppp/images/A1/plant001_rgb.png --output outputs/
python run_leaf_only_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b
```

The `vit_b` checkpoint (~375 MB) downloads to `checkpoints/` on first run and
fits a 6 GB GPU. `vit_l`/`vit_h` are more accurate but need more VRAM.

## Tunable filter thresholds

| Flag | Default | Meaning |
|---|---|---|
| `--min-shape` | `0.1` | `checkshape`: min contour-area / enclosing-circle area; drops thin/straggly blobs |
| `--subset-thresh` | `0.9` | `remove_toobig`: overlap fraction for one mask to count as contained in another (composite removal) |
| `--points-per-side` | `32` | SAM sampling grid; lower = faster / less VRAM, coarser |

Defaults are the paper's. The green-colour hue/saturation bands (`checkcolour`)
and the whole-plant IoU cutoff (`checkfullplant`) are hardcoded to match the
notebook; edit the functions if your imagery differs (e.g. non-green foliage).

## Original authors' code

This folder reimplements the method so it runs cleanly today. To use the
authors' exact code, clone it and follow their notebook:

```bash
git clone https://github.com/Dom3442/leafonlysam upstream
```

Sources: [paper](https://arxiv.org/abs/2305.09418) ·
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772375524001205) ·
[original repo](https://github.com/Dom3442/leafonlysam)
