# Leaf segmentation model zoo — runnable tests

A folder-per-model test harness for the leaf-segmentation models from the
literature review, so you can try each one on your own tree/branch imagery (or
point clouds) and compare. Every model has its own README with exact install +
run commands.

## First: which models actually apply to you?

```
                        ┌─ 2D RGB photos ─────────────────────────────────────┐
                        │                                                      │
   What data do you     │   Want it working NOW, no labels?                    │
   have?  ──────────────┤     → SAM 2 (01) · Leaf Only SAM (02) · HQ-SAM (03)  │
                        │                                                      │
                        │   Have (or will make) labels, want best accuracy?    │
                        │     → YOLO11-seg (05) · Mask R-CNN (04) · M2Former(06)│
                        │                                                      │
                        │   Subject is trees/branches outdoors?                │
                        │     → Poplar-leaf + LeafInst (08); CSIRO leaf MRCNN(07)│
                        └──────────────────────────────────────────────────────┘
                        ┌─ 3D LiDAR / TLS point clouds ───────────────────────┐
                        │   Separate foliage from wood?                        │
                        │     → PointsToWood (09, ready) · LWSNet (10) ·        │
                        │       RandLA-Net tropical (11)                       │
                        └──────────────────────────────────────────────────────┘
```

Two questions decide everything:
1. **2D images or 3D point clouds?** Completely different model families.
2. **Individual leaf instances, or just leaf-vs-wood separation?** (3D is
   almost always the latter.)

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
| 09 | `3d_leaf_wood/09_pointstowood` | **PointsToWood** | ✅ pretrained | ready 3D leaf–wood separation |
| 10 | `3d_leaf_wood/10_lwsnet` | **LWSNet** | ⚠️ train-your-own | F1≈97% reported |
| 11 | `3d_leaf_wood/11_randlanet_tropical` | **RandLA-Net (tropical)** | ⚠️ train-your-own | mIoU 86.8% on 148 trees |

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
python run_*.py --image ../../data/samples/synthetic_leaf.png --output outputs/
```

## Try it in 3 minutes (no photos, no GPU-heavy download yet)

```bash
# 1) make a throwaway synthetic branch+leaves image (needs only numpy + Pillow)
python3 -m venv .venv-data && source .venv-data/bin/activate
pip install -r requirements-common.txt
python3 data/make_synthetic_sample.py     # -> data/samples/synthetic_leaf.png
deactivate

# 2) then pick a model folder (start with 01_sam2) and follow its README
```

## Your hardware (detected: RTX 4050 Laptop, 6 GB VRAM)

6 GB is enough for every 2D model **if you pick the small variants** — the
per-folder READMEs call out which. Rules of thumb:

- SAM 2 → `--model-size tiny|small`; SAM v1 / HQ-SAM → `vit_b` / `vit_tiny`.
- Mask2Former → `swin-tiny`/`swin-small`; YOLO → `n`/`s` (train `--batch 4`).
- If you hit CUDA out-of-memory: shrink the model, add `--points-per-side 16`
  (SAM family), lower image size, or run `--device cpu` (slower but works).
- 3D point-cloud nets tile the cloud; use smaller batches for big plots.

## Recommended path

1. **Prototype zero-shot** with SAM 2 (01) or Light HQ-SAM (03) — immediate,
   no labels. Leaf Only SAM (02) if you want the foliage post-processing.
2. **Need accuracy?** Annotate a small set (or get **Poplar-leaf**, folder 08)
   and **fine-tune YOLO11-seg** (05) — best effort/accuracy trade-off. Mask R-CNN
   (04) / Mask2Former (06) are alternatives.
3. **3D LiDAR/TLS?** Go straight to **PointsToWood** (09) — pretrained, runnable.

## Layout

```
leaf_segmenter/
├── README.md                 ← you are here
├── requirements-common.txt   ← numpy+Pillow for the shared helper & sample maker
├── shared/leafviz.py         ← mask overlay / IO used by all 2D scripts
├── data/                     ← put your images / point clouds here (+ sample maker)
├── 2d_foundation/            ← 01 SAM2 · 02 Leaf Only SAM · 03 HQ-SAM
├── 2d_instance_seg/          ← 04 Mask R-CNN · 05 YOLO-seg · 06 Mask2Former
├── 2d_forestry/              ← 07 CSIRO · 08 LeafInst/Poplar
├── 3d_leaf_wood/             ← 09 PointsToWood · 10 LWSNet · 11 RandLA-Net
└── eval/                     ← leaf-count evaluation vs CVPPP ground truth
```

## Evaluate leaf counts (CVPPP LCC)

The CVPPP LCC set (`data/samples/A1` … `A4`) has a ground-truth leaf count per
image. To score a model's segmented-leaf count against it:

```bash
# 1) run a model with --count-csv (inside its venv) to record predicted counts
python run_*.py --input-dir ../../data/samples/A1 \
    --output outputs/A1 --count-csv outputs/A1/counts.csv

# 2) compare to ground truth (stdlib only; runs anywhere)
python eval/evaluate_leaf_count.py --pred outputs/A1/counts.csv \
    --gt data/samples/A1/A1.csv
```

Already ran with `--crops` but not `--count-csv`? Skip step 1 and evaluate the
crop folders directly: `--from-crops outputs/A1` instead of `--pred`.

Reports the CVPPP metrics — DiC, |DiC|, MSE, % agreement. See `eval/README.md`.
Best suited to the zero-shot foliage models (01–03); the COCO baselines (04–06)
need fine-tuning first. Evaluate one subset (A1…A4) at a time.
