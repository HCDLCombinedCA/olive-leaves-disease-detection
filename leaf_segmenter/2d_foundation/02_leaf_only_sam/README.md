# 02 — Leaf Only SAM

**What:** A zero-shot pipeline that runs the original **SAM (v1)** automatic mask
generator, then post-processes the masks to isolate leaves (size filter, border
filter, green-color filter, duplicate removal). No training/annotation needed.

**Reference numbers (from the paper, potato leaves):** recall 63.2 / precision
60.3 zero-shot, vs. 78.7 / 74.7 for a fine-tuned Mask R-CNN — i.e. a fine-tuned
model still wins, but this needs zero labels.

**Status:** ✅ Runs out of the box. `run_leaf_only_sam.py` is a faithful
reimplementation of the post-processing; SAM v1 weights auto-download.

## Install

```bash
cd 2d_foundation/02_leaf_only_sam
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python run_leaf_only_sam.py --image ../../data/samples/synthetic_leaf.png --output outputs/
python run_leaf_only_sam.py --input-dir ../../data/samples --model-type vit_b
```

The `vit_b` checkpoint (~375 MB) downloads to `checkpoints/` on first run and
fits a 6 GB GPU. `vit_l`/`vit_h` are more accurate but need more VRAM.

## Tunable filter thresholds

| Flag | Default | Meaning |
|---|---|---|
| `--min-green` | `0.5` | required green-pixel fraction per mask |
| `--min-area` / `--max-area` | `0.0005` / `0.2` | mask size as fraction of image |
| `--border-frac` | `0.15` | drop masks hugging the image edge (background/pot) |
| `--contain` | `0.8` | drop a mask mostly contained in a larger kept one |

Tune these to your imagery — the defaults suit close-ups of foliage on a
contrasting background.

## Original authors' code

This folder reimplements the method so it runs cleanly today. To use the
authors' exact code, clone it and follow their notebook:

```bash
git clone https://github.com/Dom3442/leafonlysam upstream
```

Sources: [paper](https://arxiv.org/abs/2305.09418) ·
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772375524001205) ·
[original repo](https://github.com/Dom3442/leafonlysam)
