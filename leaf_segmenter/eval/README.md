# eval/ — leaf evaluation

Three evaluators, scoring different things:

- `evaluate_leaf_count.py` — Leaf **Counting** Challenge (LCC) metrics: does the
  model report the right *number* of leaves.
- `evaluate_leaf_segmentation.py` — Leaf **Segmentation** Challenge (LSC)
  metrics: how well the predicted *masks* overlap the true leaves (FBD, SBD).
  A model can nail the count while merging/splitting leaves, or vice
  versa — these are complementary, not redundant.
- `evaluate_timings.py` — **runtime**: mean / median / max seconds per image for
  inference and post-processing, compared across models. Needs no ground truth.

The first two score against CVPPP 2017 ground truth and take the same three
flags: `--input-dir` (a model's `--output-dir`), `--gt-dir` (ground truth), and
an optional `--output-dir` (write the full per-image table). Evaluate **one
CVPPP subset (A1…A4) at a time** — they reuse `plantNNN` ids for different
plants. The timing evaluator takes one or more `--input-dir`s and the optional
`--output-dir` only.

## Data

Ground truth lives under `data/cvppp/`, one subfolder per CVPPP subset (`A1`…`A4`):

```
data/cvppp/
├─ images/A?/         plantNNN_rgb.png + A?.csv (image, leaf_count) — for LCC
├─ mask/A?/           plantNNN_fg.png            binary foreground — (see note)
└─ per_leaf_mask/A?/  plantNNN_label.png         0=bg, 1..N=leaf id — for FBD + SBD
```

The segmentation evaluator reads only the per-leaf label maps: it derives the
foreground for FBD from the label map (`label > 0`), which is pixel-identical to
the separate `_fg.png` blob, so `mask/` is no longer needed for evaluation.

`data/split_dataset.py` can carve these into `fine_tuning/` (10%) and `testing/`
(90%) folders with the same structure (images + both masks + a filtered count
CSV), so you can evaluate on a held-out split by pointing `--gt-dir` at e.g.
`testing/images/A1` (counts) or `testing/per_leaf_mask/A1` (masks).

## All evaluators read a model's `--output-dir`

Run any 2D model with `--output-dir` (inside that model's venv). Into that dir it
writes:

- `counts.csv` — `image,n_leaves` (for counting)
- `<image>/mask_000.png`, `mask_001.png`, … — one binary mask per predicted leaf,
  in a per-image folder (for segmentation)
- `timings.csv` — `image,inference_s,postprocess_s` (for runtime)

```bash
cd models/02_leaf_only_sam
python run_leaf_only_sam.py --input-dir ../../data/cvppp/images/A1 --output-dir output/A1
```

## Leaf counting (`evaluate_leaf_count.py`)

Pure stdlib — runs in any env.

```bash
python eval/evaluate_leaf_count.py \
    --input-dir models/02_leaf_only_sam/output/A1 \
    --gt-dir    data/cvppp/images/A1
```

`--input-dir` is the model's output dir (its `counts.csv` is read). `--gt-dir` is
a folder holding the CVPPP `A?.csv` count file(s). Add `--output-dir DIR` to print
the full per-image table and write it to `DIR/leaf_count_per_image.csv`.

### Metrics (CVPPP LCC)

| Metric | Meaning |
|--------|---------|
| **DiC** | mean signed difference `pred − gt` — counting **bias** (+ = over-count) |
| **\|DiC\|** | mean absolute difference — typical error magnitude |
| **MSE** | mean squared error (RMSE also shown) |
| **Agreement** | % of images counted **exactly** right |

Rows are matched by image basename, so predictions and ground truth can live in
different folders. `--gt-dir` merges every `*.csv` under it; if you accidentally
point it at multiple subsets (which reuse `plantNNN` names) it prints a WARNING.

## Leaf segmentation (`evaluate_leaf_segmentation.py`)

Scores mask *quality*, not just count. Needs numpy + Pillow
(`requirements-common.txt`) — unlike the counting evaluator, this one isn't pure
stdlib.

```bash
python eval/evaluate_leaf_segmentation.py \
    --input-dir models/02_leaf_only_sam/output/A1 \
    --gt-dir    data/cvppp/per_leaf_mask/A1
```

`--input-dir` is the model's output dir (its `<image>/mask_*.png` folders are
read). `--gt-dir` is the folder of per-leaf `plantNNN_label.png` maps. Add
`--output-dir DIR` for the full per-image FBD/SBD table (also written to
`DIR/leaf_segmentation_per_image.csv`). Images are matched by canonical plant id
(`_masks`/`_rgb`/`_fg`/`_label` and the extension stripped, so `plant001_rgb/` and
`plant001_label.png` resolve to `plant001`).

### Metrics (CVPPP LSC)

| Metric | Meaning |
|--------|---------|
| **FBD** | Dice between the union of predicted masks and the GT foreground — "did it find the plant", regardless of leaf separation |
| **SBD** | Symmetric Best Dice — mean best-match Dice per leaf, both directions, worse of the two kept. The standard CVPPP leaderboard number |

## Runtime (`evaluate_timings.py`)

Pure stdlib — runs in any env. Reads the `timings.csv` each run script writes
(one row per image, timed around the model forward pass and the mask
post-processing separately) and reports, per model, the mean, median, and max
seconds per image for inference, post-processing, and their total. Pass several
`--input-dir`s to compare models in one table:

```bash
python eval/evaluate_timings.py \
    --input-dir models/01_sam2/output/A1 \
                models/03_hq_sam/output/A1
```

Add `--output-dir DIR` to also write the table to `DIR/timing_summary.csv`. Dirs
without a `timings.csv` are skipped with a warning.

**Warm-up caveat:** the first image of a run absorbs one-off cost (CUDA kernel
loading etc.), which inflates the mean and max more than the median — on small
image sets the median is the more honest per-image figure. Timings are only
comparable when produced on the same machine and image set.

## Caveat: which models are worth evaluating now

The zero-shot foliage models — **02 Leaf Only SAM** (built for exactly this),
**01 SAM2**, **03 HQ-SAM** — produce meaningful leaf counts out of the box.

The COCO-pretrained instance models (**04 Mask R-CNN, 05 YOLO-seg,
06 Mask2Former**) have no "leaf" class and will mostly predict ~0 leaves until
fine-tuned, so their metrics here are only meaningful **after** training on a leaf
dataset (see `05_yolo_seg/train_yolo_seg.py`).
