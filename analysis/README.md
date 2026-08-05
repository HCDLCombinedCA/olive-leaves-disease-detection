# Analysis: corrections and bias investigation

Standalone work on defects and gaps found in the existing components. **Nothing
in `glassbox/`, `leaf_segmenter/` or the root notebooks is modified** — each item
here is a parallel implementation the team can review and merge.

Runs on `kquille/hcaim_gpu_tudublin_course:latest` unmodified; `run.sh` installs
nothing.

```bash
./run.sh python glassbox_fixed.py
./run.sh python acquisition_bias.py
./run.sh python gradcam_report.py --model ../compression/artifacts/densenet121_baseline
```

---

## 1. `glassbox_fixed.py` — three defects in the glass-box pipeline

Adapted from `glassbox/Olive_Leaf_GlassBox.ipynb`. Feature extraction is
reproduced faithfully (same colour fractions, same GLCM parameters) so results
stay comparable.

**Filename/label misalignment in the exported CSV.** The notebook shuffles `X`
and `Y` but not `paths`, then builds the dataframe from the unshuffled `paths`
and the shuffled labels. 269 of the first 400 rows (67%) carry the wrong
filename — `A163_leaf00.png` is listed as `olive_peacock_spot` though it comes
from `Healthy/`. Model training is unaffected, since features and labels stay
paired, but the exported `leaf_features.csv` cannot be used to trace a prediction
back to an image. Fixed by not shuffling: both splitters shuffle internally, so
the manual shuffle served no purpose. Verified — **0 mismatches** in the
regenerated CSV.

**Group leakage across the split.** The segmenter emits one crop per leaf, so a
photograph can yield `_leaf00`, `_leaf01`, … Of 3,003 crops, 550 (18.3%) come
from the 267 photographs that produced more than one, and a plain
`train_test_split` scatters those siblings across both sides. Replaced with
`StratifiedGroupKFold` grouped on the source-image stem, which also restores the
stratification the original call omitted. Verified — **0 source photographs
shared** between train and test.

**Only one classical model.** The assignment requires two from its fixed list.
Logistic Regression is added, and its standardised coefficients are reported as
the glass-box explanation the rubric asks for.

Results on the corrected grouped split (2,008 train / 995 test crops):

| model | accuracy | macro-F1 |
|---|---|---|
| Decision Tree (`max_depth=3`) | 0.624 | 0.621 |
| **Logistic Regression** | **0.711** | **0.692** |

Logistic Regression outperforms the tree by 7 points of macro-F1, so it is worth
reporting as the stronger glass-box baseline rather than an afterthought. Its
coefficients read cleanly in agronomic terms: `frac_yellow` (chlorosis) is the
dominant positive weight for `olive_peacock_spot` (+1.98) and the dominant
negative one for `Healthy` (−1.81).

One observation for the report: the GLCM texture features are computed over the
whole crop rather than inside the leaf mask, so they also describe background
texture. Given how strongly capture source correlates with class here (see
below), that is worth stating as a limitation.

---

## 2. `acquisition_bias.py` — is the model learning the camera?

`review.txt` §3.6.3 flagged background/camera bias as a risk that was never
tested. Tested here two independent ways.

### Part A — the confound is real, and severe in training

Filenames encode the capture device. Cross-tabulated against class in the
training split:

| capture source | n | dominant class | purity |
|---|---|---|---|
| bare numeric (`1.jpg` …) | 690 | `aculus_olearius` | **100.0%** |
| `B-*` camera roll | 610 | `Healthy` | **99.8%** |
| `A*` camera roll | 250 | `olive_peacock_spot` | 88.8% |
| `IMG_*` phone | 1151 | `olive_peacock_spot` | 83.6% |
| `DSC_*` DSLR | 19 | `olive_peacock_spot` | 78.9% |

A classifier given **nothing but the filename prefix**:

```
train accuracy            0.918
test  accuracy            0.499
majority-class baseline   0.382
```

So a model that never looks at a leaf can reach **91.8% on training data**. Any
training accuracy on this dataset must be read with that in mind. The official
test split partially breaks the confound — its source mix differs — which is why
the same rule only reaches 49.9% there, still 12 points above the majority
baseline.

### Part B — but the model does not exploit it

Grad-CAM++ attention measured inside versus outside a leaf mask (the HSV rule
from the team's own glassbox feature extractor, so both components agree on what
counts as leaf). 75 test images, 25 per class, randomly sampled — not the first N
alphabetically, since filenames are ordered by capture source and that would have
drawn every `Healthy` sample from one camera.

A ratio above 1 means more attention on background than an even spread would give:

| subset | n | attention outside leaf | background area | ratio |
|---|---|---|---|---|
| all | 75 | 10.9% | 15.5% | **0.70** |
| correct predictions | 72 | 10.8% | 15.3% | 0.71 |
| incorrect predictions | 3 | 13.0% | 19.3% | 0.67 |

Attention concentrates on leaf tissue well beyond chance, and errors are not
explained by background reliance. **The confound exists in the data but this
model is not using it on average** — a result worth stating explicitly, because
it is the kind of claim that is usually asserted rather than measured.

Measured on the MobileNetV2 baseline trained on the de-leaked split
(`compression/`), test macro-F1 0.930.

#### Two caveats that belong in the report

**Grad-CAM resolution.** The last convolutional layer of DenseNet121 outputs
7×7 spatial cells, upsampled to 224×224 for display. The heatmaps are therefore
coarse blobs and **cannot localise individual lesions** — the strongest claim the
evidence supports is "attends to leaf tissue", not "attends to disease lesions".
The same coarseness blurs attention across the leaf boundary, which inflates the
measured share falling outside the mask. That bias works *against* the finding,
so the 0.70 ratio is a conservative estimate of how leaf-focused the model is.

**A single error traces directly to the confound.** `a245.jpg` (true class
`aculus_olearius`) is misclassified as `olive_peacock_spot` with 0.979
confidence. Its filename prefix is `a*`, and in the training split the `A*`
capture source is **88.8% `olive_peacock_spot`**; only 9 of the 200 test images
for this class share that prefix, and its background is a grey-green surface
rather than the white paper used elsewhere. So while the model does not lean on
background *on average*, it is misled on individual images whose capture context
is rare for their class. That is a more useful statement for the report than
either "it uses background" or "it does not".

---

## 3. `gradcam_report.py` — explanations on a model that works

Replaces the saliency section of `Olive_Leaf_CNN_Model_Notebook_withSaliency.ipynb`.
Four problems with the original:

1. **The explained model is the broken one.** Cells 31–32 save `modelBest`, the
   scratch CNN, and cell 35 loads it back, so every heatmap in cells 39–45
   describes a model with macro-F1 0.401 whose `aculus_olearius` recall is 0.5%.
   Confirmed by reading the architecture out of the committed `1785117388.zip`.
   The DenseNet121 that reached 0.946 was never saved.
2. **Heatmaps use the true label, not the predicted one.** For a class the model
   never predicts, that maps a pathway which does not fire.
3. **Only correct predictions**, three per class, taken alphabetically. The
   rubric asks for at least two correct *and* two incorrect per class.
4. **No annotations** — no predicted class, true class, or confidence.

This script scores the whole test split first, then deliberately samples correct
and incorrect cases per class. Each panel is annotated with the true class, the
predicted class and its confidence; for a misclassification it additionally shows
the map for the true class alongside, which is what makes the error readable.
