# Leaf Segmentation Model Comparison — CVPPP A1

**Date:** 2026-07-22
**Dataset:** CVPPP `testing/A1` split — 115 images, 1,867 ground-truth leaf instances (~16.2 leaves/image)
**Evaluated:** 11 model configurations across 5 model families
**Evaluators:** `evaluate_leaf_count.py` (LCC), `evaluate_leaf_segmentation.py` (LSC), `evaluate_timings.py`

All three evaluators were run against each model's `output/<variant>/A1` directory, with
ground truth from `data/cvppp/testing/images/A1` (counts) and
`data/cvppp/testing/per_leaf_mask/A1` (masks).

---

## TL;DR

- **Fine-tuning is everything.** The COCO-pretrained instance models (Mask R-CNN, YOLO,
  Mask2Former) have no "leaf" class and find essentially nothing out of the box —
  SBD ≈ 0, under-counting by ~14 leaves/image. After fine-tuning they jump to the top.
- **Best overall: fine-tuned YOLO11-seg and fine-tuned Mask R-CNN**, essentially tied on
  quality (SBD 0.847 vs 0.832, FBD 0.958 vs 0.952). **YOLO wins decisively on speed** —
  0.015 s/image vs 0.068 s, and ~250× faster than any SAM model.
- **Best with no labels: SAM2 (base-plus)** — FBD 0.85, SBD 0.60 straight out of the box, no
  training. The trade-off is speed (~3.9 s/image) and a persistent ~4-leaf under-count.
- **Leaf Only SAM underperforms** its plain SAM2 cousins here (SBD 0.34), despite being the
  leaf-specific method — its aggressive post-filters appear to over-prune this data.
- **Mask2Former** was only available as COCO baselines (no fine-tuned variant) and, as
  expected, scores near-zero SBD — it is not usable here until fine-tuned.

**Practical recommendation:** fine-tuned **YOLO11-seg** for production (top accuracy + fastest);
fine-tuned **Mask R-CNN** if you need the most calibrated leaf *count*; **SAM2 base-plus** when
you have no labels at all.

---

## Headline comparison

Best configuration per family (↑ = higher better, ↓ = lower better):

| Model (best variant) | FBD ↑ | SBD ↑ | \|DiC\| ↓ | Speed (s/img) ↓ | Needs labels? |
|---|---|---|---|---|---|
| **YOLO11-seg (fine-tuned)** | **0.958** | **0.847** | 1.98 | **0.015** | yes |
| **Mask R-CNN (fine-tuned)** | 0.952 | 0.832 | **1.24** | 0.068 | yes |
| SAM2 (base-plus) | 0.845 | 0.601 | 4.59 | 3.884 | **no** |
| Leaf Only SAM (vit_b) | 0.519 | 0.336 | 10.70 | 8.903 | no |
| Mask2Former (COCO only) | 0.401 | 0.040 | 14.18 | 0.035 | *needs fine-tune* |

---

## What the metrics mean

The three headline metrics answer three *different* questions, and a model can do
well on one while failing another — which is exactly why all three are reported.
In one line: **DiC** = did it get the *number* of leaves right, **FBD** = did it
find the *plant*, **SBD** = did it get the *individual leaves* right.

Two of them (FBD, SBD) are built on the **Dice coefficient**, the standard overlap
score between two binary masks A and B:

> Dice(A, B) = 2 · |A ∩ B| / (|A| + |B|)

It runs from **0** (no overlap) to **1** (identical), and is just the intersection
counted twice over the total area of both masks.

### DiC — Difference in Count *(leaf counting, `evaluate_leaf_count.py`)*

`DiC = predicted_leaf_count − ground_truth_leaf_count`, averaged over all images.
It measures **counting bias**:

- **Negative** → the model *under-counts* (finds fewer leaves than there are);
  **positive** → it *over-counts*; **≈ 0** → on average it is well calibrated.
- Because it is *signed*, positive and negative errors on different images
  **cancel out** — a DiC near zero means "unbiased on average", *not* "correct on
  every image". That is why the report also shows **|DiC|** (mean *absolute* count
  error, which does not cancel) and **Agreement** (% of images counted *exactly*
  right). Example: a model that is +3 on one image and −3 on the next has DiC = 0
  but |DiC| = 3.
- It says nothing about mask *quality* — a model can count the right number of
  leaves while drawing them in completely the wrong places.

### FBD — Foreground-Background Dice *(did it find the plant?)*

The Dice overlap between the **union of all predicted masks** and the **union of
all ground-truth leaves** (the whole-plant silhouette). It asks *"did the model
find the plant at all?"* and is **blind to how the plant is split into leaves**:

- **High FBD (→ 1)** → the predicted masks cover the real plant pixels well.
- A model that outputs **one big blob** over the whole plant still scores high FBD,
  even though it found zero *individual* leaves. So FBD is a floor, not the goal.
- Low FBD means the model is missing the plant outright (or hallucinating masks off
  the plant) — as the COCO baselines do here.

### SBD — Symmetric Best Dice *(did it get the individual leaves right?)*

The instance-level score, and the **standard CVPPP leaderboard number**. It grades
how well *each* predicted leaf matches *one* real leaf:

- For every ground-truth leaf, take its **best Dice** against any predicted mask,
  and average over all GT leaves — that is "best Dice" in one direction. Do the
  same the other way (each predicted leaf → best GT leaf). **SBD keeps the *worse*
  (minimum) of the two directions** — that is the "symmetric" part.
- Keeping the worse direction is what makes it **penalise both mistakes**:
  *merging* several leaves into one blob (hurts the GT→pred direction) *and*
  *splitting* one leaf into many fragments (hurts the pred→GT direction).
- **This is the metric that actually reflects per-leaf segmentation quality.** The
  gap between a model's FBD and SBD is diagnostic: high FBD + low SBD (e.g. SAM2 at
  0.85 / 0.60) means it located the plant but under-segmented it into too few
  leaves.

**Reading them together:** FBD ≥ SBD in practice (finding the plant is easier than
splitting it into the right leaves). DiC tells you the *count* is right; SBD tells
you the *masks* are right; a good model needs both, since either can be satisfied
while the other fails.

---

## 1. Leaf counting (`evaluate_leaf_count.py`)

CVPPP LCC metrics. **DiC** = mean signed error (pred − gt), so negative = under-counting;
**|DiC|** = mean absolute error; **RMSE** = root mean squared error; **Agree** = % of images
counted *exactly* right.

| Model | DiC (bias) | \|DiC\| | RMSE | Agreement |
|---|---|---|---|---|
| SAM2 (tiny) | −6.29 | 6.30 | 6.75 | 0.0% |
| SAM2 (small) | −4.39 | 4.63 | 5.26 | 2.6% |
| SAM2 (base-plus) | −4.31 | 4.59 | 5.15 | 6.1% |
| Leaf Only SAM (vit_b) | −6.49 | 10.70 | 11.61 | 1.7% |
| Mask R-CNN (COCO) | −13.72 | 13.72 | 13.86 | 0.0% |
| **Mask R-CNN (fine-tuned)** | **−0.01** | **1.24** | **1.70** | **29.6%** |
| YOLO11-seg (COCO) | −14.45 | 14.45 | 14.58 | 0.0% |
| YOLO11-seg (fine-tuned) | +1.50 | 1.98 | 2.50 | 15.7% |
| Mask2Former (swin-tiny) | −14.18 | 14.18 | 14.33 | 0.0% |
| Mask2Former (swin-small) | −14.82 | 14.82 | 14.95 | 0.0% |
| Mask2Former (swin-base) | −15.50 | 15.50 | 15.62 | 0.0% |

**Read-out:** Fine-tuned Mask R-CNN is almost perfectly calibrated (DiC −0.01, i.e. on
average it neither over- nor under-counts) and gets **30% of images exactly right**.
Fine-tuned YOLO slightly *over*-counts (+1.5). Every zero-shot / COCO model **under-counts** —
the SAM family mildly (~4–6 leaves short, it misses small/occluded leaves), the COCO baselines
catastrophically (~14 short — they find almost no leaves at all).

---

## 2. Mask quality (`evaluate_leaf_segmentation.py`)

CVPPP LSC metrics. **FBD** = Foreground-Background Dice ("did it find the plant at all",
regardless of how leaves are split); **SBD** = Symmetric Best Dice, the standard CVPPP
leaf-splitting score (per-leaf best-match Dice, worse direction kept).

| Model | FBD ↑ | SBD ↑ |
|---|---|---|
| SAM2 (tiny) | 0.825 | 0.547 |
| SAM2 (small) | 0.848 | 0.592 |
| SAM2 (base-plus) | 0.845 | 0.601 |
| Leaf Only SAM (vit_b) | 0.519 | 0.336 |
| Mask R-CNN (COCO) | 0.417 | 0.059 |
| **Mask R-CNN (fine-tuned)** | 0.952 | 0.832 |
| YOLO11-seg (COCO) | 0.281 | 0.032 |
| **YOLO11-seg (fine-tuned)** | **0.958** | **0.847** |
| Mask2Former (swin-tiny) | 0.401 | 0.040 |
| Mask2Former (swin-small) | 0.455 | 0.035 |
| Mask2Former (swin-base) | 0.225 | 0.016 |

**Read-out:**
- **Fine-tuned YOLO and Mask R-CNN dominate** — FBD ~0.95, SBD ~0.84. Both locate the plant
  almost perfectly *and* split it into the right leaves. YOLO edges both metrics; the two are
  otherwise neck-and-neck.
- **SAM2 is the strongest zero-shot option**, improving with model size
  (SBD 0.547 → 0.592 → 0.601 for tiny → small → base-plus). Note the gap between its high FBD
  (~0.85) and middling SBD (~0.60): it locates the plant well but under-segments it into too
  few leaves — the same under-count seen in the LCC metrics.
- **Leaf Only SAM lags** every SAM2 variant on both FBD (0.519) and SBD (0.336) — its four
  leaf-filters discard too many true leaves here.
- **COCO baselines are unusable**: SBD ≤ 0.06 across Mask R-CNN, YOLO, and all Mask2Former
  sizes. FBD is a little higher (0.28–0.46) only because the odd stray "potted plant"-type mask
  overlaps some foreground; at the leaf level there is essentially no correct segmentation. This
  is the expected "no leaf class" failure, not a bug.

---

## 3. Speed (`evaluate_timings.py`)

Median seconds per image (median is robust to first-image warm-up). Inference = model forward
pass; post = mask filtering/formatting.

| Model | Inference (med) | Post-proc (med) | Total (med) | Throughput |
|---|---|---|---|---|
| **YOLO11-seg (fine-tuned)** | **0.015** | 0.031 | 0.046 | ~22 img/s |
| YOLO11-seg (COCO) | 0.015 | 0.004 | 0.019 | ~53 img/s |
| Mask2Former (swin-tiny) | 0.035 | 0.014 | 0.049 | ~20 img/s |
| Mask2Former (swin-small) | 0.043 | 0.013 | 0.056 | ~18 img/s |
| Mask2Former (swin-base) | 0.053 | 0.014 | 0.067 | ~15 img/s |
| Mask R-CNN (COCO / fine-tuned) | 0.068 | ~0.005 | 0.071 / 0.074 | ~14 img/s |
| SAM2 (tiny) | 3.772 | 0.009 | 3.781 | ~0.26 img/s |
| SAM2 (small) | 3.804 | 0.012 | 3.816 | ~0.26 img/s |
| SAM2 (base-plus) | 3.884 | 0.015 | 3.899 | ~0.26 img/s |
| Leaf Only SAM (vit_b) | 8.903 | 0.045 | 8.948 | ~0.11 img/s |

**Read-out:** The instance models are **50–250× faster** than the SAM family. SAM is slow by
design — its automatic mask generator runs a 32×32 grid of prompts (~1,024 forward passes per
image); Leaf Only SAM adds crop layers and contour post-filters on top, making it the slowest.
Post-processing is negligible everywhere except relative to YOLO's tiny inference time.

---

## Discussion

**Zero-shot vs fine-tuned.** The gap is stark. With no labels, SAM2 base-plus reaches SBD 0.60.
With ~13 fine-tuning images per class, YOLO and Mask R-CNN reach SBD ~0.84 — a **+0.24 SBD**
jump — while also running two orders of magnitude faster. If you can label even a small set,
fine-tuning a lightweight instance model is by far the better path.

**YOLO vs Mask R-CNN (the two winners).** They are close enough that the choice is about
priorities:
- **YOLO11-seg** — highest FBD and SBD, and *far* the fastest. Best default for throughput
  and mask quality. Slightly over-counts (DiC +1.5).
- **Mask R-CNN** — a hair behind on mask quality, but the best count calibration (DiC −0.01)
  and the most images counted exactly right. Pick it when an accurate leaf *count* matters
  most and 4× slower is acceptable.

**SAM family.** Bigger is better (tiny < small < base-plus) but with diminishing returns and
rising cost. All share the same profile: high FBD but lower SBD — the leaves they output are
clean, but they systematically miss small and occluded ones (hence the under-count). Leaf Only
SAM's specialised filters are a net negative on this dataset.

**Mask2Former.** Only COCO baselines were generated, so it cannot be judged fairly here — all
three sizes score near-zero SBD, exactly as the "no leaf class" caveat predicts. Curiously,
larger backbones gave *lower* FBD (more, larger spurious masks). A fine-tuned Mask2Former is the
obvious missing experiment.

---

## Caveats

- **Single subset.** All numbers are CVPPP `testing/A1` (115 images). Other subsets (A2–A4) may
  differ; evaluate one subset at a time (they reuse `plantNNN` names).
- **Timings are only comparable on the same machine.** They are reported as regenerated
  together; treat them as relative, not absolute. SAM's cost is inherent to its grid-prompt
  design, not a hardware artifact.
- **HQ-SAM (folder 03) was not evaluated** — no output directory was present.
- **No fine-tuned Mask2Former** was available; its poor numbers reflect COCO weights only.
- **Count agreement (% exact)** is a deliberately harsh metric — being off by one leaf scores as
  a miss.

---

## Reproduce

```bash
# ground truth: testing split, subset A1
GT_COUNT=data/cvppp/testing/images/A1
GT_SEG=data/cvppp/testing/per_leaf_mask/A1
OUT=models/05_yolo_seg/output/fine-tuned/A1     # any model output dir

python eval/evaluate_leaf_count.py        --input-dir $OUT --gt-dir $GT_COUNT
python eval/evaluate_leaf_segmentation.py --input-dir $OUT --gt-dir $GT_SEG
python eval/evaluate_timings.py           --input-dir $OUT   # accepts several dirs at once
```
