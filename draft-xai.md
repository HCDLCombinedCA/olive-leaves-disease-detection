# XAI scope, dimensions, and model drift

Draft for the two XAI sections the plan requires but which no deliverable
currently covers: item 22 (define the scope and dimensions of XAI before
applying methods) and item 27 (model drift as a post-hoc monitoring concern).

Numbers come from `analysis/results/`, `compression/results/` and
`analysis/README.md`.

---

## 1. Why define the scope first

Explanations are not a single thing. A heatmap that satisfies a marker will not
satisfy a farmer deciding whether to spray, and neither answers the question a
developer asks when a model regresses. Choosing methods before deciding which
question is being answered is how projects end up with a figure that looks like
XAI but supports no claim. This section fixes the question first; the methods
follow from it.

Three audiences are involved, and they want different things.

| audience | question they are asking | what answers it |
|---|---|---|
| technical assessor | is the model reasoning from disease evidence rather than an artefact? | attention localisation, measured against a leaf mask |
| agronomist or grower | why this leaf, and can I act on it? | a local explanation on a single image, with the predicted class and confidence |
| developer or maintainer | what broke, and has behaviour shifted since deployment? | per-class recall, confidence distribution, and attention patterns tracked over time |

## 2. The dimensions used, and what each one bought

### 2.1 Local against global

**Local** explanations dominate this work: Grad-CAM++ and LIME both explain a
single prediction. This is the right default for a diagnostic tool, since the
user is always looking at one leaf, and it is what the rubric asks for — at least
two correct and two incorrect predictions per class, each annotated with the
predicted class, the true class and the confidence.

**Global** behaviour was obtained by aggregating local explanations rather than
by a separate global method. Grad-CAM attention was integrated inside and outside
a leaf mask across 75 sampled test images, giving a single number — the share of
attention falling on background — that describes the model rather than an image.
That aggregate is what makes the claim "the model attends to leaf tissue"
checkable instead of anecdotal.

The glass-box models supply global explanation directly: the logistic regression
coefficient table is a complete description of the model, not a summary of it.

### 2.2 Model-specific against model-agnostic

**Grad-CAM++ is model-specific.** It needs the gradients and the final
convolutional feature map, so it applies to the CNNs and to nothing else. It is
cheap and gives spatially precise output, within the resolution limits noted
below.

**LIME is model-agnostic.** It needs only a function from images to class
probabilities. That property was not a preference here but a requirement: a
`.tflite` model is an inference-only graph with no gradients, so Grad-CAM cannot
run on any of the compressed variants. Comparing explanations before and after
compression is only possible with a model-agnostic method, and the same procedure
then runs unchanged against the Keras baseline and every quantised variant.

This is worth stating explicitly in the report, because it is a case where the
choice of XAI method was forced by a deployment decision rather than chosen for
its own sake.

### 2.3 Intrinsic (ante-hoc) against post-hoc

**Intrinsic.** The decision tree and logistic regression are interpretable by
construction. Their explanations are not approximations of the model — they are
the model. The tree at depth 3 has seven decision nodes and can be read aloud;
the logistic regression coefficients translate into agronomic vocabulary because
the features were designed that way (`frac_yellow` is a chlorosis proxy,
`frac_dark` necrosis, `frac_gray` silvering).

**Post-hoc.** Grad-CAM++ and LIME are applied to trained black-box models and
approximate their behaviour. They are the only option for the CNNs, and the price
is that an explanation can be wrong about the model in ways that are hard to
detect.

The comparison is itself a result: on the same official 680-image test split, the
best intrinsic model reaches macro-F1 0.742 and the post-hoc-explained CNNs reach
0.939. That gap of roughly 20 points is the cost of interpretability on this
task, and it is the number the trade-off discussion should be built on. It has to
be quoted from the common protocol on the best feature set: the segmented-crop
figure of 0.831 is measured on held-out crops from training photographs, and the
nine base features reach only 0.667, so both would misstate the gap.

### 2.4 Purpose

Four distinct uses, each of which produced something:

| purpose | what it produced |
|---|---|
| **debugging** | revealed that the saliency maps in the original notebook explain a model with macro-F1 0.401 whose recall on `aculus_olearius` is 0.5% — the working model had never been saved |
| **bias detection** | attention measured against a leaf mask, testing whether the model uses the capture-source confound |
| **trust calibration** | high-confidence errors identified and inspected; the model's mistakes are between the two diseases, not between diseased and healthy |
| **monitoring** | the attention and confidence measures give a baseline against which post-deployment drift can be detected — see section 4 |

## 3. What the explanations actually showed

**Errors are between diseases, not between diseased and healthy.** Across the
annotated Grad-CAM panels, both high-confidence mistakes are
`aculus_olearius` ↔ `olive_peacock_spot` confusions at 0.979 and 0.991
confidence; none is a diseased leaf called healthy. For an agricultural decision
support tool this is the benign failure direction — a missed diagnosis is the
dangerous one — and it should be stated because it is not obvious from macro-F1.

**On average the model does not use the background.** Grad-CAM++ attention
integrated inside and outside a leaf mask, over 75 randomly sampled test images:

| subset | n | attention outside leaf | background area | ratio |
|---|---|---|---|---|
| all | 75 | 10.9% | 15.5% | **0.70** |
| correct predictions | 72 | 10.8% | 15.3% | 0.71 |
| incorrect predictions | 3 | 13.0% | 19.3% | 0.67 |

A ratio below 1 means attention concentrates on leaf tissue more than an even
spread would produce. Errors are not explained by background reliance.

**But individual failures do trace to it.** `a245.jpg`, a true
`aculus_olearius`, is classified as `olive_peacock_spot` at 0.979 confidence. Its
filename prefix is `a*`; in training the `A*` capture source is 88.8%
`olive_peacock_spot`, only 9 of the 200 test images for this class share that
prefix, and its background is a grey-green surface rather than the white paper
used elsewhere. The honest summary is three-layered: the confound exists and is
severe, the model does not rely on it on average, and it is nonetheless misled on
images whose capture context is rare for their class.

**Compression preserves the reasoning, in proportion to the accuracy it costs.**
LIME explanations before and after quantisation, using a fixed segmentation and
a fixed random seed so any difference is attributable to the models:

| variant | mean Spearman ρ | mean top-5 Jaccard | label agreement |
|---|---|---|---|
| float16 | 0.599 | 0.833 | 100% (6/6) |
| dynamic range | 0.556 | 0.548 | 83% (5/6) |
| full integer | 0.407 | 0.627 | 100% (6/6) |

The ordering follows the accuracy loss. Read the two metrics together: a rank
correlation of 0.4–0.6 says the ordering of evidence shifts, while the higher
top-5 overlap says the evidence the prediction actually rests on is stable and
the instability sits in the low-weight regions.

Label agreement is not uniform: dynamic range changed the predicted class on one
of the six images, the other two variants on none. With six images no rate should
be read into that, which is why the counts are given raw.

### Limits on these claims

Grad-CAM output for DenseNet121 comes from a 7×7 feature grid upsampled to
224×224. The heatmaps are coarse blobs and **cannot localise individual lesions**;
the strongest supportable claim is attention to leaf tissue. The same coarseness
blurs attention across the leaf boundary and therefore inflates the measured
share falling outside it, which makes the 0.70 ratio a conservative estimate
rather than an optimistic one. LIME fidelity rests on six images at 500
perturbations each — the ordering across variants is consistent, but the absolute
correlations are a small sample.

---

## 4. Model drift

Drift is where post-hoc XAI stops being a reporting exercise and becomes an
operational one. The useful question is not whether this model will drift, but
which signals would reveal it and what the monitoring thresholds should be.

### 4.1 Sources of drift, ranked by what the data says

The dataset makes some of these measurable rather than hypothetical.

**Capture device — the largest and best-evidenced risk.** The training set
contains five distinct capture sources, and they are almost perfectly confounded
with class. A model deployed on a phone model absent from training sees an input
distribution the training set never covered. This is not speculative: the
`B-*` camera roll differs from the phone images by a factor of 64 in brown
fraction within the same class, and a filename-only classifier reaches 91.8% on
training data. `a245.jpg` is a worked example of the resulting failure.

**Background and setting.** Every training image is a detached leaf on a plain
surface — white paper, a wooden table. Deployment on leaves photographed *in
situ*, against soil, sky or other foliage, is a distribution the model has never
seen. The leaf-mask heuristic used throughout this project also assumes a plain
background and would fail alongside the model.

**Season and phenology.** Symptom appearance changes through the growing season.
The dataset's timestamps run from February to August 2019 — one partial season,
one year. Nothing in the data constrains behaviour in autumn or across years.

**Cultivar and geography.** Leaf morphology and colour vary by olive cultivar,
and the dataset does not record cultivar at all. Neither is geography recorded.
Generalisation beyond the unknown origin region is untested by construction.

**Image pipeline.** Every published image is exactly 800×600, so a resizing or
compression step was applied before release. A deployment pipeline with different
JPEG quality or resizing is already a shift, and a subtle one: the
119 duplicate pairs found in this dataset were mostly invisible to a byte hash
precisely because re-encoding changes bytes without changing pixels.

**New or co-occurring conditions.** The label set is three classes. A leaf with a
condition outside that set — or with two conditions at once — will be forced into
one of the three, confidently. Nothing in the model can express "not one of
these".

### 4.2 What to monitor

Five signals, in increasing order of how much they cost to collect.

| signal | detects | how it would move |
|---|---|---|
| **predicted class distribution** | input shift, silently | a jump in the `olive_peacock_spot` share with no agronomic explanation |
| **confidence distribution** | out-of-distribution input | the mean rising (over-confident on unfamiliar input) or the distribution flattening toward 0.33 |
| **per-class recall on a labelled audit sample** | genuine degradation | the minority class first — `aculus_olearius` was the class both compression and the original scratch CNN damaged first |
| **attention outside the leaf mask** | the model turning to background | the 0.70 baseline ratio rising toward or past 1.0 |
| **LIME top-5 region stability on a fixed probe set** | reasoning shifting without accuracy moving yet | overlap with the reference explanations falling below the 0.83 seen under lossless quantisation |

The last two are the contribution this project can make to the monitoring
argument, because both have measured baselines rather than invented thresholds.
A ratio of 0.70 and a top-5 Jaccard of 0.83 are what this model does now; a
deployed system can be compared against those numbers.

### 4.3 How drift would show up as harm

Two failure modes matter, and they are asymmetric.

**False negatives — a diseased leaf called healthy — allow untreated spread.**
The current model does not make this error on the sampled panels, but the
mechanism by which it would start to is clear: under-representation of a
capture source at deployment shifts the decision boundary, and the minority class
degrades first. Per-class recall on `aculus_olearius`, not overall accuracy, is
the signal that would move.

**Background-driven predictions are the harder failure**, because accuracy can
stay flat while the reasoning becomes invalid. A model that has learned "wooden
table means healthy" performs perfectly until someone photographs a diseased leaf
on a wooden table. Accuracy monitoring cannot see this coming; attention
monitoring can, which is the argument for tracking the leaf-mask ratio rather
than metrics alone.

### 4.4 What follows for deployment

* **Log the capture device.** The single strongest predictor of failure in this
  dataset is which camera took the picture, and it costs nothing to record.
* **Hold out a labelled audit set per capture source**, not just per class, so
  cross-device degradation is visible before users report it.
* **Keep a fixed probe set of images** with reference Grad-CAM and LIME
  explanations, and re-run them on every model update. Explanation stability
  catches changes that accuracy misses.
* **Do not deploy as an autonomous decision.** The failure that matters is a
  missed diagnosis; expert confirmation before treatment decisions is the control
  that bounds it. This connects directly to the ethics section.
