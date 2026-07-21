# eval/ — leaf evaluation

Two evaluators, scoring different things against CVPPP 2017 ground truth:

- `evaluate_leaf_count.py` — Leaf **Counting** Challenge (LCC) metrics: does the
  model report the right *number* of leaves.
- `evaluate_leaf_segmentation.py` — Leaf **Segmentation** Challenge (LSC)
  metrics: how well the predicted *masks* overlap the true leaves (FBD, SBD,
  AP). A model can nail the count while merging/splitting leaves, or vice
  versa — these are complementary, not redundant.

## Data

Ground truth lives under `data/cvppp/`, one subfolder per CVPPP subset
(`A1`…`A4`):

```
data/cvppp/
├─ images/A?/         plantNNN_rgb.png + A?.csv (image, leaf_count) — for LCC
├─ mask/A?/           plantNNN_fg.png            binary foreground/background — for FBD
└─ per_leaf_mask/A?/  plantNNN_label.png         0=bg, 1..N=leaf id — for SBD/AP
```

`mask/` and `per_leaf_mask/` are two *different* annotations from the same
CVPPP release — don't confuse them. `_fg.png` has no leaf identities (just one
foreground blob), so it can only support FBD. `_label.png` has one integer per
leaf, so it's what SBD and AP actually match against.

## Leaf counting (`evaluate_leaf_count.py`)

Because every model runs in its own virtualenv, evaluation is decoupled from
inference:

**1. Run a model with `--count-csv`** (inside that model's venv) to write its
predicted count per image. Example with Leaf Only SAM on A1:

```bash
cd 2d_foundation/02_leaf_only_sam
python run_leaf_only_sam.py \
    --input-dir ../../data/cvppp/images/A1 \
    --output outputs/A1 \
    --count-csv outputs/A1/counts.csv
```

This writes `outputs/A1/counts.csv` with rows `image,n_leaves` (plus the usual
overlay PNGs). `--count-csv` works on **all** 2D run scripts (01–06).

**2. Evaluate** the predictions against ground truth (stdlib only — runs in any
env):

```bash
python eval/evaluate_leaf_count.py \
    --pred 2d_foundation/02_leaf_only_sam/outputs/A1/counts.csv \
    --gt   data/cvppp/images/A1/A1.csv
```

Add `--per-image` for the full table and `--out results.csv` to save per-image
diffs.

### Already ran with `--crops`?

If you produced crop folders (`--crops`) but no `--count-csv` file, skip step 1 —
the number of `leaf_*.png` cutouts in each `<image>_leaves/` folder is the
predicted count. Point the evaluator at the output dir instead:

```bash
python eval/evaluate_leaf_count.py \
    --from-crops 2d_foundation/01_sam2/outputs/A1 \
    --gt         data/cvppp/images/A1/A1.csv
```

**Evaluate one subset at a time.** `--gt` also accepts a *folder* (merges every
`*.csv` under it), but the A1/A2/A3 subsets reuse the same `plantNNN_rgb.png`
names for different plants, so merging them by name is ambiguous — the evaluator
prints a WARNING if it detects this. Run A1, A2, A3, A4 separately.

### Metrics (CVPPP LCC)

| Metric | Meaning |
|--------|---------|
| **DiC** | mean signed difference `pred − gt` — counting **bias** (+ = over-count) |
| **\|DiC\|** | mean absolute difference — typical error magnitude |
| **MSE** | mean squared error (RMSE also shown) |
| **Agreement** | % of images counted **exactly** right |

Rows are matched by image basename, so predictions and ground truth can live in
different folders.

## Leaf segmentation (`evaluate_leaf_segmentation.py`)

Scores mask *quality*, not just count. Needs numpy + Pillow (`requirements-common.txt`)
— unlike the counting evaluator, this one isn't pure stdlib.

**1. Run a model with `--masks`** to write one full-size binary PNG per
predicted leaf (white=leaf, black=background):

```bash
cd 2d_foundation/02_leaf_only_sam
python run_leaf_only_sam.py \
    --input-dir ../../data/cvppp/images/A1 \
    --output outputs/A1 \
    --masks
```

This writes `outputs/A1/<image>_masks/mask_000.png`, `mask_001.png`, ... per image.
`--masks` works on all 2D run scripts (01–06), same as `--crops`/`--count-csv`.

**2. Evaluate** against the ground-truth foreground and/or per-leaf label masks:

```bash
python eval/evaluate_leaf_segmentation.py \
    --pred-masks 2d_foundation/02_leaf_only_sam/outputs/A1 \
    --gt-fg      data/cvppp/mask/A1 \
    --gt-label   data/cvppp/per_leaf_mask/A1 \
    --per-image --out results.csv
```

`--gt-fg` and `--gt-label` are independent — pass either alone to get just the
metrics it supports (FBD only, or SBD+AP only), or both for the full set.
Images are matched by canonical plant id, stripping `_masks`/`_rgb`/`_fg`/`_label`
and the extension, so `plant001_rgb_masks/`, `plant001_fg.png` and
`plant001_label.png` all resolve to `plant001`. Same one-subset-at-a-time caveat
as counting applies (A1/A2/A3 reuse `plantNNN` ids).

### Metrics (CVPPP LSC)

| Metric | Meaning | Needs |
|--------|---------|-------|
| **FBD** | Dice between the union of predicted masks and the GT foreground blob — "did it find the plant", regardless of leaf separation | `--gt-fg` |
| **SBD** | Symmetric Best Dice — mean best-match Dice per leaf, in both directions, worse of the two kept. The standard CVPPP leaderboard number | `--gt-label` |
| **AP@t** (default t=0.5) | Precision/Recall/F1 from a greedy one-to-one IoU match between predicted and GT leaf instances | `--gt-label` |

**AP caveat:** `--masks` output has no per-instance confidence score, so this is
a single IoU-threshold operating point (P/R/F1 @ IoU≥t), not a confidence-ranked,
multi-threshold COCO-style mAP. SBD is far more forgiving of a merged/split leaf
(partial Dice credit) than AP (binary hit/miss at the threshold) — expect SBD to
look noticeably better than AP·F1 for the same predictions when models
under-segment touching leaves.

## Caveat: which models are worth evaluating now

The zero-shot foliage models — **02 Leaf Only SAM** (built for exactly this),
**01 SAM2**, **03 HQ-SAM** — produce meaningful leaf counts out of the box.

The COCO-pretrained instance models (**04 Mask R-CNN, 05 YOLO-seg,
06 Mask2Former**) have no "leaf" class and will mostly predict ~0 leaves until
fine-tuned, so their metrics here are only meaningful **after** training on a leaf
dataset (see `05_yolo_seg/train_yolo_seg.py`).
