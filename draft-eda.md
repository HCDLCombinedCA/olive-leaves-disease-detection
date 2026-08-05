# Data exploration and preprocessing

Draft for the data-exploration section. Figures and numbers come from
`analysis/dataset_audit.py`, `analysis/acquisition_bias.py` and
`compression/src/prepare_data.py`; regenerate with those scripts.

This covers items 8–14 of the project plan: counts, integrity, dimensions, class
balance, duplicates, bias, and the preprocessing decisions that follow from them.

---

## 1. What the dataset contains

3,400 images across three classes, with a train/test split supplied by the
publisher.

| class | train | test | total | share |
|---|---|---|---|---|
| Healthy | 830 | 220 | 1,050 | 30.9% |
| aculus_olearius | 690 | 200 | 890 | 26.2% |
| olive_peacock_spot | 1,200 | 260 | 1,460 | 42.9% |
| **total** | **2,720** | **680** | **3,400** | |

The imbalance is mild — the majority class is 1.64× the minority — but it runs
the same way in both splits, so a model that simply favours
`olive_peacock_spot` scores 42.9% before learning anything. This is the baseline
against which accuracy has to be read, and the reason macro-F1 and per-class
recall are reported throughout rather than accuracy alone. Class weights were
used in training for the same reason.

## 2. Integrity and format

| check | result |
|---|---|
| corrupt or unreadable files | **0** of 3,400 |
| file format | JPEG, 3,400 of 3,400 |
| colour mode | RGB, 3,400 of 3,400 |
| distinct image dimensions | **1** — every image is 800×600 |
| file size | 14.0 KB min, 28.9 KB median, 55.5 KB max |

Nothing is broken, and no format handling is required. The striking result is
that **every image is exactly 800×600**. Photographs taken on a phone, a DSLR and
at least two other cameras do not naturally share a resolution, so the dataset
has been resized before publication. Two consequences follow.

First, no aspect-ratio or resizing policy needs to be decided; a single
224×224 resize is uniform across the set.

Second, and less comfortably, the images are not raw captures. Whatever
processing produced a uniform 800×600 has already been applied, which limits how
far conclusions here transfer to a phone photographing a leaf in a grove. The
model is trained on curated images, and that belongs in the limitations
discussion.

## 3. Duplicates and near-duplicates

The published split leaks. Detection by byte hash finds 24 images present in both
splits, and that figure is misleading: of the 100 train/test pairs sharing a
filename, only 19 are byte-identical, while the other 81 have different MD5 sums
yet are pixel-identical at 800×600 (mean absolute error 0.0000 between 32×32
thumbnails). They are the same photograph re-encoded. Comparing 16×16 greyscale
thumbnails instead of bytes raises the count to **119 duplicate pairs covering
114 training images**.

Separately, filenames of the form `IMG_YYYYMMDD_HHMMSS.jpg` carry capture
timestamps running in consecutive seconds — burst sequences of a single leaf.
**52.6% of the test images for `olive_peacock_spot`** have a training image taken
within ten seconds. No hash detects these, because the camera moves slightly
between frames, but to a convolutional network they are images it has seen.

| leak type | training images affected |
|---|---|
| duplicated in test | 114 |
| shares a photo burst with a test image | 264 |
| **removed from training** | **359** |

The official test split was left intact and removal applied to the training side
only, so results stay comparable with other work on this dataset.

**Evidence this mattered.** On the original split, the DenseNet121 transfer model
reported validation macro-F1 0.819 against test 0.946 — a held-out test set
scoring 13 points *above* validation, which should not happen. After
de-duplication and a group-aware split the same architecture gives 0.9237 against
0.9387, a gap of 1.5 points. The anomaly disappears, which is the clearest
indication it was leakage rather than chance.

## 4. Colour and intensity by class

Mean statistics over a seeded random sample of 400 images per class, measured on
the whole frame:

| class | mean intensity | mean saturation | green | yellow | brown |
|---|---|---|---|---|---|
| Healthy | 0.731 | 0.114 | 0.067 | 0.006 | **0.172** |
| aculus_olearius | 0.747 | 0.086 | 0.090 | 0.017 | 0.008 |
| olive_peacock_spot | 0.721 | 0.097 | 0.042 | 0.045 | 0.017 |

Two things stand out, and they point in opposite directions.

The **yellow fraction rises with disease** — 0.006 for healthy leaves, 0.017 for
`aculus_olearius`, 0.045 for `olive_peacock_spot`. Yellowing is the visible sign
of chlorosis, so this is the signal a classifier ought to use, and it is the
feature the glass-box logistic regression weights most heavily
(+1.98 for `olive_peacock_spot`, −1.81 for `Healthy`).

The **brown fraction is twenty times higher for healthy leaves** than for either
diseased class, which makes no agronomic sense. Browning is a symptom, not a sign
of health. Splitting the same statistic by capture device explains it:

| capture source / class | n sampled | mean intensity | brown fraction |
|---|---|---|---|
| **B\* / Healthy** | 265 | 0.724 | **0.255** |
| IMG_ / Healthy | 125 | 0.745 | 0.004 |
| IMG_ / aculus_olearius | 69 | 0.741 | 0.006 |
| IMG_ / olive_peacock_spot | 325 | 0.731 | 0.009 |
| numeric / aculus_olearius | 307 | 0.749 | 0.006 |
| A\* / olive_peacock_spot | 68 | 0.669 | 0.015 |
| DSC_ / aculus_olearius | 17 | 0.755 | 0.064 |

The `B-*` camera roll carries a brown-dominated background — a wooden surface —
and it supplies 73% of the training `Healthy` images. Within the same class,
switching camera changes the brown fraction from 0.255 to 0.004, a factor of 64.
The class-level figure is not describing leaves at all; it is describing one
photographer's table.

This is worth stating carefully, because it is the strongest single argument in
the bias investigation and it was found by accident. A colour statistic that
looked like a class difference turned out to be a device difference, and it was
only visible once the same statistic was split by capture source.

## 5. Capture source predicts the label

The filename prefix encodes the device. Cross-tabulated against class in the
training split:

| capture source | n (train) | dominant class | purity |
|---|---|---|---|
| bare numeric (`1.jpg` …) | 690 | `aculus_olearius` | **100.0%** |
| `B-*` camera roll | 610 | `Healthy` | **99.8%** |
| `A*` camera roll | 250 | `olive_peacock_spot` | 88.8% |
| `IMG_*` phone | 1,151 | `olive_peacock_spot` | 83.6% |
| `DSC_*` DSLR | 19 | `olive_peacock_spot` | 78.9% |

Every one of the 690 training images for `aculus_olearius` has a bare numeric
filename, and no image of any other class does. A classifier given nothing but
the filename prefix reaches **91.8% accuracy on the training split**, against a
38.2% majority baseline.

Any training accuracy on this dataset therefore has to be read with the
possibility that the model is recognising the photographer rather than the
disease. The official test split partially breaks the correlation — its source
mix differs — and the same rule reaches only 49.9% there, still 12 points above
the baseline.

Whether the trained model actually exploits the shortcut is measured separately
in the XAI section; the short answer is that on average it does not, but it fails
on individual images whose capture context is unusual for their class.

## 6. Preprocessing decisions and their justification

| decision | reason |
|---|---|
| resize to 224×224, keep colour | required by the ImageNet backbones; colour is the primary disease signal (chlorosis, necrosis, silvering) so greyscale is not an option |
| ImageNet normalisation per backbone | MobileNetV2 expects [−1, 1] and DenseNet uses channel normalisation; pairing them wrongly silently destroys accuracy |
| augmentation on training only | rotation ±15°, zoom 0.1, shift 0.1, horizontal and vertical flip. Leaves have no canonical orientation, so flips and rotations are label-preserving |
| no augmentation on validation or test | so their scores reflect the real input distribution |
| balanced class weights | mild imbalance, and the minority class is the one the original scratch CNN failed to learn at all |
| **stratified, group-aware train/validation split** | see below |

The split deserves its own note. The original notebook used
`ImageDataGenerator(validation_split=0.3)`, whose documented behaviour is to take
the **last 30% of each class's sorted filenames**. Since the filename prefix
encodes the capture device, and the archive is ordered by prefix, that default
does not produce a random sample: for `Healthy` it puts roughly 76% phone images
in validation while training sees 95% `B-*` camera images. The validation set
became an unintended cross-camera test, which is why validation scores sat so far
below test scores.

The replacement splits at the level of *leaf identity* — bursts and duplicate
groups merged by union-find — and stratifies by class. Verified: 1,505 groups,
none straddling the boundary, class proportions matching across splits to within
0.1 percentage points.

| class | original train | after de-duplication | train | val | test |
|---|---|---|---|---|---|
| Healthy | 830 | 723 | 578 | 145 | 220 |
| aculus_olearius | 690 | 684 | 547 | 137 | 200 |
| olive_peacock_spot | 1,200 | 954 | 763 | 191 | 260 |
| **total** | **2,720** | **2,361** | **1,888** | **473** | **680** |

## 7. What the exploration changed

Three findings altered the experimental design rather than merely describing the
data:

1. **The leak forced a rebuild of the training set.** Reporting the 0.946 figure
   from the original split would have overstated performance; the corrected
   figure is 0.9387, and more importantly the validation/test relationship became
   interpretable.
2. **The split method was wrong, not just the split.** Fixing the leakage without
   fixing `validation_split` would have left validation measuring cross-camera
   generalisation while appearing to measure ordinary held-out accuracy.
3. **Capture source is a confound that has to be tested, not assumed away.** It
   is strong enough to reach 91.8% on training data by itself, so the XAI work
   had to include a direct measurement of whether the model uses it.

## 8. Suggested figures

Available or straightforward to generate from the audit output:

* class distribution bar chart (`counts` in `analysis/results/dataset_audit.json`)
* sample image grid, one row per class — already in the CNN notebook
* brown-fraction bar chart grouped by capture source, which is the single most
  persuasive figure in this section
* the source × class cross-tabulation as a heatmap

An image-size distribution plot is not worth a figure: every image is 800×600, so
it is a single bar. The finding is better stated in one sentence.
