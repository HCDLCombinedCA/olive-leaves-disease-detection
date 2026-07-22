# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

A **folder-per-model test harness for 2D leaf instance segmentation** from
photos of trees/branches. Goal: try each model from the literature on the
user's own imagery and compare. Not a training pipeline for a single model —
it's a zoo of runnable evaluators. (A `3d_leaf_wood/` folder for LiDAR/TLS
leaf-vs-wood point classification existed earlier and was removed — different
task, out of scope here.)

## Hardware target (important)

Development machine: **WSL2, Python 3.12, NVIDIA RTX 4050 Laptop, 6 GB VRAM**, no
conda. Every default in this repo assumes 6 GB — pick small model variants. If a
model OOMs: shrink the variant, `--points-per-side 16` (SAM family), lower image
size, or `--device cpu`.

## Layout

```
README.md                    master guide: decision tree, status table, workflow
requirements-common.txt      numpy+Pillow (shared helper + evaluate_leaf_segmentation.py)
shared/helper.py            mask overlay / image IO used by ALL 2d scripts
data/cvppp/                  CVPPP sample images + ground truth (images/mask/per_leaf_mask)
2d_foundation/               01 SAM2 · 02 Leaf Only SAM · 03 HQ-SAM   (zero-shot)
2d_instance_seg/             04 Mask R-CNN · 05 YOLO-seg · 06 Mask2Former
2d_forestry/                 07 CSIRO (legacy TF1) · 08 LeafInst/Poplar (no code yet)
eval/                        leaf-count + leaf-mask eval vs CVPPP ground truth
```

## Leaf evaluation

Two evaluators against `data/cvppp/` ground truth (`A1`…`A4` subsets). First run
a model with `--output-dir DIR` (writes `DIR/counts.csv` and per-image
`DIR/<image>/mask_*.png` folders, via `shared.helper.save_outputs`); both
evaluators then take the same `--input-dir DIR` / `--gt-dir` / optional
`--output-dir` (the last writes the full per-image table).

- `eval/evaluate_leaf_count.py` scores a model's segmented-leaf **count**
  (CVPPP LCC). Reads `<input-dir>/counts.csv` (`image,n_leaves`, count =
  `len(masks)`) and merges the ground-truth `A?.csv` files under `--gt-dir`
  (e.g. `data/cvppp/images/A1`). Metrics: DiC, |DiC|, MSE, % agreement. Pure
  stdlib.
- `eval/evaluate_leaf_segmentation.py` scores mask *overlap quality* (CVPPP LSC):
  FBD, SBD, and AP@IoU. Reads the predicted `<input-dir>/<image>/mask_*.png`
  folders and the per-leaf label maps under `--gt-dir`
  (`data/cvppp/per_leaf_mask/A?/plantNNN_label.png`); the FBD foreground is
  derived from those maps (`label > 0`, pixel-identical to `mask/A?/*_fg.png`),
  so no separate foreground dir is needed. Needs numpy+Pillow (not stdlib-only).
  AP is a single IoU-threshold (`IOU_THRESH`, default 0.5) precision/recall/F1,
  not a confidence-ranked mAP curve — the mask PNGs have no per-instance score.

Both match by image basename/canonical id, ignoring extension — **evaluate one
subset at a time**, since A1/A2/A3 reuse `plantNNN_rgb.png` names.
`data/split_dataset.py` (run inside `data/cvppp/`) carves a random 10%
`fine_tuning/` + 90% `testing/` split with the same layout and a filtered count
CSV.

## Core conventions

- **One virtualenv per model folder.** The stacks (`sam2`, `segment-anything`,
  `ultralytics`, `transformers`, `torch-geometric`, TF1) have conflicting
  dependencies — never merge them into one env. Each folder has its own
  `requirements.txt`.
- **Every 2D run script shares the same CLI shape:** `--input-dir` (required), a
  single model selector (`--model-size`/`--model-type`/`--weights`/`--model`),
  and an optional `--output-dir`; all other knobs are module-level constants.
  Each exposes a `run()` that returns per-image leaf crops and, with
  `--output-dir`, writes a per-image folder (overlay + `mask_*.png`) plus
  `counts.csv` via `shared.helper.save_outputs`.
- **Scripts import the shared helper** via `sys.path.insert(0, REPO_ROOT)` then
  `from shared.helper import ...`. `shared/helper.py` must stay dependency-light
  (numpy + Pillow only) so it imports inside every venv.
- **Weights auto-download where possible** (SAM2, HQ-SAM via HF hub, SAM v1 via
  URL, YOLO/Mask2Former via their libs). Checkpoints/venvs/outputs are
  gitignored.

## Key caveat to remember

Folders **04 (Mask R-CNN), 05 (YOLO), 06 (Mask2Former)** load **COCO-pretrained**
weights, which have **no "leaf" class**. Out of the box they are runnable
baselines that find little on bare branches — their value is as fine-tuning
backbones. Real leaf results there require training (see `05_yolo_seg`'s
`train_yolo_seg.py` + `leaf.yaml`). The zero-shot models (01–03) produce output
with no training.

## Status quick-reference

- ✅ runnable now: 01, 02, 03, 04, 05, 06 (04–06 = baselines pending fine-tune)
- ⚠️ extra setup: 07 (needs Python 3.7 + TF1.15)
- ⏳ no public code yet: 08 (LeafInst/Poplar — March 2026 paper; use 05 as fallback)

## Verifying changes

Fast checks that need no heavy deps:
- `find . -name '*.py' -not -path '*/.venv*/*' | xargs -I{} python3 -m py_compile {}`
- `bash -n <script>.sh` for setup scripts
- Smoke-test the shared helper: make a venv with `requirements-common.txt`, run
  `python3 -c "from shared.helper import load_image, overlay_masks, save_image, write_masks; ..."`
  against a real sample under `data/cvppp/images/A1/`.

Actually running a model requires installing that folder's `requirements.txt`
(multi-GB torch + model weights) — do it inside that folder's venv.
