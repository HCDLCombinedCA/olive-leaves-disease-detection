# Results

Generated from the artefacts in `compression/results/` and `analysis/results/`. Regenerate with `python3 make_report_tables.py`.

## 1. Dataset preparation

Leakage was detected with a 16x16 greyscale thumbnail, mean absolute error < 0.020. MD5 alone finds only 24 of the 119 duplicate pairs -- the remainder are the same photograph re-encoded, identical in pixels but not in bytes.

| leak type | training images removed |
|---|---|
| duplicated in test | 114 |
| shares a burst with a test image | 264 |
| **total** | **359** |

| class | original | after de-dup | train | val | test |
|---|---|---|---|---|---|
| Healthy | 830 | 723 | 578 | 145 | 220 |
| aculus_olearius | 690 | 684 | 547 | 137 | 200 |
| olive_peacock_spot | 1200 | 954 | 763 | 191 | 260 |
| **total** | **2720** | **2361** | **1888** | **473** | **680** |

The official test split is left intact; removal is on the training side only, so results stay comparable with other work on this dataset.

---

## 2. Leaf segmentation

Model selection is documented in `leaf_segmenter/eval/evaluation_report.md`: eleven configurations across five families benchmarked on CVPPP A1, won by fine-tuned YOLO11-seg (SBD 0.847, FBD 0.958, 0.015 s/image). That model was then applied to the olive photographs.

| leaves detected | photographs | share |
|---|---|---|
| 0 | 326 | 12.0% |
| 1 | 2127 | 78.2% |
| 2 | 252 | 9.3% |
| 3 | 14 | 0.5% |
| 4 | 1 | 0.0% |
| **total** | **2720** | |

The crop set reconciles exactly: **3003 files = 2677 detected leaves + 326 whole-image fallbacks**. Where the segmenter found nothing, the pipeline writes the entire photograph as `<stem>_full.png`, so 11% of the "leaf crops" feeding the glass-box features are unsegmented photographs with their background intact -- relevant given the capture-source confound in section 8.

---

## 3. Deep models: from scratch against transfer learning

| model | parameters | train time | val macro-F1 | test macro-F1 | test recall Healthy | test recall Aculus | test recall Peacock |
|---|---|---|---|---|---|---|---|
| scratch CNN (notebook architecture) | 5,767,139 | 2687 s | 0.7765 | 0.8095 | 0.927 | 0.735 | 0.773 |
| scratch CNN (redesigned) | 585,059 | 2632 s | 0.9162 | 0.8712 | 0.814 | 0.830 | 0.950 |
| MobileNetV2 (ImageNet) | 2,261,827 | 1175 s | 0.9142 | 0.9296 | 0.950 | 0.885 | 0.950 |
| DenseNet121 (ImageNet) | 7,040,579 | 3460 s | 0.9237 | 0.9387 | 0.895 | 0.970 | 0.954 |

- **scratch CNN (notebook architecture)**: val 0.777 vs test 0.810 (gap +0.033).
- **scratch CNN (redesigned)**: val 0.916 vs test 0.871 (gap -0.045).
- **MobileNetV2 (ImageNet)**: val 0.914 vs test 0.930 (gap +0.015).
- **DenseNet121 (ImageNet)**: val 0.924 vs test 0.939 (gap +0.015).

The original notebook reported val 0.819 against test 0.946. A validation score far *below* the test score is the inversion that first suggested leakage, and no model here reproduces it.

Worth a sentence in the report: **scratch CNN (redesigned)** is the only model whose validation score *overstates* its test score. Validation is carved from the training photographs while the test split is the dataset's own, so a model that leans on whatever the training photographs share -- capture conditions included, see section 8 -- will look better on validation than it is. Transfer learning does not show this, which is consistent with its features coming from ImageNet rather than from these photographs.

Every model here shares `compression/src/common.py`: the same manifests, input size, augmentation, class weights and metrics. The scratch entries differ from the transfer entries only in initialisation and architecture, and the first scratch entry reproduces the notebook's chosen architecture unchanged, so the effect of the corrected data is separated from the effect of the design.

---

## 4. Glass-box models

| protocol | model | test images | accuracy | macro-F1 |
|---|---|---|---|---|
| segmented | Decision Tree (depth 3) | 995 | 0.624 | 0.621 |
| segmented | Logistic Regression | 995 | 0.711 | 0.692 |
| common | Decision Tree (depth 3) | 680 | 0.684 | 0.667 |
| common | Logistic Regression | 680 | 0.621 | 0.600 |

- **segmented**: 2008 train / 995 test, StratifiedGroupKFold on the source photograph, 0 shared source photographs.
- **common**: 2361 train / 680 test, official test split from compression/data/prepared manifests, 0 shared source photographs.

The two protocols do not rank the models the same way, which is worth stating rather than smoothing over: the tree gains on whole photographs while logistic regression loses. The engineered colour fractions are computed inside a leaf mask that a tight crop makes reliable and a full photograph does not, and a linear model has no way to compensate for that where a depth-3 tree's thresholds partly can.

Individual predictions -- two correct and two incorrect per model, with the decision path or the exact per-feature contribution behind each -- are in `analysis/results/glassbox_*_examples.png` and in the `explanations` key of each JSON.

---

## 5. RQ a: like-for-like comparison

All rows below are scored on the **same official 680-image test split**. The segmented-crop glass-box numbers in section 4 are measured on a different population and are deliberately not carried into this table.

| model | accuracy | macro-F1 | explanation |
|---|---|---|---|
| scratch CNN (notebook architecture) | 0.812 | 0.810 | post-hoc (Grad-CAM / LIME) |
| scratch CNN (redesigned) | 0.871 | 0.871 | post-hoc (Grad-CAM / LIME) |
| MobileNetV2 (ImageNet) | 0.931 | 0.930 | post-hoc (Grad-CAM / LIME) |
| DenseNet121 (ImageNet) | 0.940 | 0.939 | post-hoc (Grad-CAM / LIME) |
| Decision Tree (depth 3) (engineered features) | 0.684 | 0.667 | intrinsic (exact) |
| Logistic Regression (engineered features) | 0.621 | 0.600 | intrinsic (exact) |

---

## 6. Compression

### mobilenetv2

| variant | size (MB) | gzip (MB) | latency (ms) | img/s | test macro-F1 | recall Aculus |
|---|---|---|---|---|---|---|
| baseline | 13.211 | 8.738 | 53.55 | 18.7 | 0.9296 | 0.885 |
| pruned50 | 13.24 | 5.54 | 51.69 | 19.3 | 0.7057 | 0.69 |
| sweep20 | 13.24 | 7.619 | 51.98 | 19.2 | 0.9201 | 0.86 |
| sweep35 | 13.24 | 6.636 | 51.93 | 19.3 | 0.8892 | 0.895 |
| sweep50 | 13.24 | 5.545 | 52.1 | 19.2 | 0.8328 | 0.77 |
| sweep65 | 13.24 | 4.397 | 52.75 | 19.0 | 0.5343 | 0.725 |
| baseline_dynamic_range | 2.512 | 2.159 | 18.63 | 53.7 | 0.9341 | 0.89 |
| baseline_float16 | 4.479 | 4.113 | 11.16 | 89.6 | 0.9282 | 0.885 |
| baseline_float32 | 8.882 | 8.234 | 11.56 | 86.5 | 0.9296 | 0.885 |
| baseline_full_integer | 2.715 | 2.21 | 9.48 | 105.5 | 0.9242 | 0.875 |
| pruned50_dynamic_range | 2.512 | 1.606 | 18.72 | 53.4 | 0.7059 | 0.635 |
| pruned50_float16 | 4.479 | 2.809 | 11.52 | 86.8 | 0.7057 | 0.69 |

Baseline for reference: 13.211 MB, 53.55 ms, macro-F1 0.9296.

### densenet121

| variant | size (MB) | gzip (MB) | latency (ms) | img/s | test macro-F1 | recall Aculus |
|---|---|---|---|---|---|---|
| baseline | 38.148 | 27.061 | 119.58 | 8.4 | 0.9387 | 0.97 |
| pruned50 | 38.212 | 16.519 | 118.32 | 8.5 | 0.9474 | 0.965 |
| sweep50 | 38.212 | 16.526 | 119.1 | 8.4 | 0.945 | 0.99 |
| sweep65 | 38.229 | 12.924 | 119.2 | 8.4 | 0.9344 | 0.955 |
| sweep80 | 38.229 | 8.971 | 118.31 | 8.5 | 0.8291 | 0.81 |
| sweep90 | 38.229 | 6.139 | 119.47 | 8.4 | 0.6674 | 0.745 |
| baseline_dynamic_range | 7.421 | 6.009 | 106.66 | 9.4 | 0.9315 | 0.97 |
| baseline_float16 | 14.061 | 12.943 | 109.72 | 9.1 | 0.9387 | 0.97 |
| baseline_float32 | 27.915 | 26.08 | 110.28 | 9.1 | 0.9387 | 0.97 |
| baseline_full_integer | 7.362 | 5.858 | 57.47 | 17.4 | 0.8676 | 0.855 |
| pruned50_dynamic_range | 7.421 | 4.667 | 106.57 | 9.4 | 0.9489 | 0.975 |
| pruned50_float16 | 14.061 | 8.683 | 111.73 | 8.9 | 0.9474 | 0.965 |

Baseline for reference: 38.148 MB, 119.58 ms, macro-F1 0.9387.

Unstructured pruning zeroes weights without changing tensor shapes, so the raw file size is unchanged and the saving appears only in the gzipped column. Reporting both is what makes the pruning result readable.

---

## 7. Explanation fidelity after compression (LIME)

| variant | images | mean Spearman | mean top-5 Jaccard | label agreement |
|---|---|---|---|---|
| dynamic_range | 6 | 0.556 | 0.548 | 83% |
| float16 | 6 | 0.599 | 0.833 | 100% |
| full_integer | 6 | 0.407 | 0.627 | 100% |

LIME rather than Grad-CAM because a `.tflite` model has no gradients; a model-agnostic method lets the identical procedure run against the Keras baseline and every quantised variant.

---

## 8. Acquisition bias

### Is capture source entangled with the label?

| capture source | n (train) | dominant class | purity |
|---|---|---|---|
| bare numeric | 690 | aculus_olearius | 100.0% |
| B* (camera roll) | 610 | Healthy | 99.8% |
| A* (camera roll) | 250 | olive_peacock_spot | 88.8% |
| IMG_ (phone, timestamped) | 1151 | olive_peacock_spot | 83.6% |
| DSC_ (DSLR) | 19 | olive_peacock_spot | 78.9% |

A classifier given only the filename prefix reaches **91.8% on training data** and 49.9% on the official test split, against a 38.2% majority baseline. Training accuracy on this dataset has to be read with that in mind.

### Does the model exploit it?

Grad-CAM++ attention inside versus outside a leaf mask. A ratio above 1 means more attention on background than an even spread would give.

| subset | n | attention outside leaf | background area | ratio |
|---|---|---|---|---|
| all | 75 | 10.9% | 15.5% | **0.70** |
| correct predictions | 72 | -- | -- | 0.71 |
| incorrect predictions | 3 | -- | -- | 0.67 |

The confound is present in the data but this model is not using it: attention concentrates on leaf tissue well beyond chance, and errors are not explained by background reliance.

---
