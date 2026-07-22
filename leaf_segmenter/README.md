# Leaf segmentation model zoo — runnable tests

A folder-per-model test harness for the leaf-segmentation models from the
literature review, so you can try each one on your own tree/branch imagery and
compare. Every model has its own README with exact install + run commands.

## First: which models actually apply to you?

```
                        ┌─ 2D RGB photos ─────────────────────────────────────┐
                        │                                                      │
   What do you          │   Want it working NOW, no labels?                    │
   need?  ──────────────┤     → SAM 2 (01) · Leaf Only SAM (02) · HQ-SAM (03)  │
                        │                                                      │
                        │   Have (or will make) labels, want best accuracy?    │
                        │     → YOLO11-seg (05) · Mask R-CNN (04) · M2Former(06)│
                        │                                                      │
                        │   Subject is trees/branches outdoors?                │
                        │     → Poplar-leaf + LeafInst (08); CSIRO leaf MRCNN(07)│
                        └──────────────────────────────────────────────────────┘
```

This repo covers **2D RGB individual-leaf instance segmentation** only. (An
earlier `3d_leaf_wood/` folder tried LiDAR/TLS leaf-vs-wood separation models —
removed since that's a different task, foliage-vs-wood point classification
rather than segmenting individual leaves, and out of scope here.)

## Folder map & status

| # | Folder | Model | Runs today? | Notes |
|---|--------|-------|-------------|-------|
| 01 | `2d_foundation/01_sam2` | **SAM 2** (Meta) | ✅ zero-shot | weights auto-download; best quick start |
| 02 | `2d_foundation/02_leaf_only_sam` | **Leaf Only SAM** | ✅ zero-shot | SAM v1 + leaf post-processing |
| 03 | `2d_foundation/03_hq_sam` | **HQ-SAM** | ✅ zero-shot | sharper edges; `vit_tiny` = Light HQ-SAM |
| 04 | `2d_instance_seg/04_mask_rcnn` | **Mask R-CNN** | ✅ baseline | COCO weights; fine-tune for leaves |
| 05 | `2d_instance_seg/05_yolo_seg` | **YOLOv8/11-seg** | ✅ baseline + train | best accuracy-for-effort once labelled |
| 06 | `2d_instance_seg/06_mask2former` | **Mask2Former** | ✅ baseline | SOTA on CVPPP; fine-tune for leaves |
| 07 | `2d_forestry/07_csiro_leaf_segmenter` | **CSIRO leaf MRCNN** | ⚠️ legacy TF1 | pretrained on synth Arabidopsis; Py3.7 env |
| 08 | `2d_forestry/08_leafinst_poplar` | **LeafInst + Poplar-leaf** | ⏳ no code yet | most on-point for trees; use YOLO fallback |

✅ = pip install + run · ⚠️ = extra setup / training · ⏳ = code not released yet

## Setup model — one virtualenv per folder

These stacks (`sam2`, `segment-anything`, `ultralytics`, `transformers`,
`torch-geometric`, TF1…) have **conflicting dependencies**, so each folder gets
its own `.venv`. Don't try to install everything into one environment.

Standard pattern (from inside any `✅` folder):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python run_*.py --input-dir ../../data/cvppp/images/A1 --output-dir output/A1
```

## Try it in 3 minutes

`data/cvppp/` already ships real CVPPP sample images (see [Evaluate leaf
segmentation](#evaluate-leaf-segmentation-cvppp-lsc--lcc) below) — just pick a
model folder (start with `01_sam2`) and follow its README against
`data/cvppp/images/A1/plant001_rgb.png` or the whole `A1` folder.

## Your hardware (detected: RTX 4050 Laptop, 6 GB VRAM)

6 GB is enough for every 2D model **if you pick the small variants** — the
per-folder READMEs call out which. Rules of thumb:

- SAM 2 → `--model-size tiny|small`; SAM v1 / HQ-SAM → `vit_b` / `vit_tiny`.
- Mask2Former → `swin-tiny`/`swin-small`; YOLO → `n`/`s` (train `--batch 4`).
- If you hit CUDA out-of-memory: shrink the model, add `--points-per-side 16`
  (SAM family), lower image size, or run `--device cpu` (slower but works).

## Recommended path

1. **Prototype zero-shot** with SAM 2 (01) or Light HQ-SAM (03) — immediate,
   no labels. Leaf Only SAM (02) if you want the foliage post-processing.
2. **Need accuracy?** Annotate a small set (or get **Poplar-leaf**, folder 08)
   and **fine-tune YOLO11-seg** (05) — best effort/accuracy trade-off. Mask R-CNN
   (04) / Mask2Former (06) are alternatives.

## Layout

```
leaf_segmenter/
├── README.md                 ← you are here
├── requirements-common.txt   ← numpy+Pillow for the shared helper
├── shared/helper.py         ← mask overlay / IO used by all 2D scripts
├── data/cvppp/                ← CVPPP sample images + ground truth (see eval/README.md)
├── 2d_foundation/            ← 01 SAM2 · 02 Leaf Only SAM · 03 HQ-SAM
├── 2d_instance_seg/          ← 04 Mask R-CNN · 05 YOLO-seg · 06 Mask2Former
├── 2d_forestry/              ← 07 CSIRO · 08 LeafInst/Poplar
└── eval/                     ← leaf count / mask quality / runtime evaluation
```

## Evaluate leaf segmentation (CVPPP LSC + LCC)

`data/cvppp/` has real CVPPP ground truth: leaf **counts** (`images/A?/A?.csv`)
and per-leaf **instance** label maps (`per_leaf_mask/A?/plantNNN_label.png`).
First run a model with `--output-dir` (it writes `counts.csv`, `timings.csv`, and
per-image `<image>/mask_*.png` folders), then point the evaluators at that dir.
The two accuracy evaluators take the same `--input-dir` / `--gt-dir` / optional
`--output-dir`; the timing evaluator needs no ground truth and accepts several
`--input-dir`s to compare models side by side:

```bash
# run a model, writing predictions to output/A1
python 2d_foundation/01_sam2/run_sam2.py \
    --input-dir data/cvppp/images/A1 --output-dir output/A1

# counts (LCC)
python eval/evaluate_leaf_count.py \
    --input-dir output/A1 --gt-dir data/cvppp/images/A1

# mask quality — FBD / SBD / AP (LSC)
python eval/evaluate_leaf_segmentation.py \
    --input-dir output/A1 --gt-dir data/cvppp/per_leaf_mask/A1

# runtime — mean / median / max seconds per image
python eval/evaluate_timings.py \
    --input-dir output/A1 another_model/output/A1
```

Add `--output-dir DIR` to any of them for the full per-image / summary table.
Accuracy evaluation is best suited to the zero-shot foliage models (01–03); the
COCO baselines (04–06) need fine-tuning first. Evaluate one subset (A1…A4) at a
time. Full details, caveats, and metric definitions in `eval/README.md`.
