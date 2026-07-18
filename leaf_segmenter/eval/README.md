# eval/ — leaf-count evaluation

Compare the number of leaves a model **segments** against the **ground-truth**
leaf count for each image, using the CVPPP Leaf Counting Challenge metrics.

## Data

Ground truth is the CVPPP 2017 LCC set at `data/samples/A1` … `A4` — each folder
holds `plantNNN_rgb.png` images and an `A?.csv` of `image, leaf_count` (no header).

## Two-step workflow

Because every model runs in its own virtualenv, evaluation is decoupled from
inference:

**1. Run a model with `--count-csv`** (inside that model's venv) to write its
predicted count per image. Example with Leaf Only SAM on A1:

```bash
cd 2d_foundation/02_leaf_only_sam
python run_leaf_only_sam.py \
    --input-dir ../../data/samples/A1 \
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
    --gt   data/samples/A1/A1.csv
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
    --gt         data/samples/A1/A1.csv
```

**Evaluate one subset at a time.** `--gt` also accepts a *folder* (merges every
`*.csv` under it), but the A1/A2/A3 subsets reuse the same `plantNNN_rgb.png`
names for different plants, so merging them by name is ambiguous — the evaluator
prints a WARNING if it detects this. Run A1, A2, A3, A4 separately.

## Metrics (CVPPP LCC)

| Metric | Meaning |
|--------|---------|
| **DiC** | mean signed difference `pred − gt` — counting **bias** (+ = over-count) |
| **\|DiC\|** | mean absolute difference — typical error magnitude |
| **MSE** | mean squared error (RMSE also shown) |
| **Agreement** | % of images counted **exactly** right |

Rows are matched by image basename, so predictions and ground truth can live in
different folders.

## Caveat: which models are worth evaluating now

The zero-shot foliage models — **02 Leaf Only SAM** (built for exactly this),
**01 SAM2**, **03 HQ-SAM** — produce meaningful leaf counts out of the box.

The COCO-pretrained instance models (**04 Mask R-CNN, 05 YOLO-seg,
06 Mask2Former**) have no "leaf" class and will mostly predict ~0 leaves until
fine-tuned, so their metrics here are only meaningful **after** training on a leaf
dataset (see `05_yolo_seg/train_yolo_seg.py`).
