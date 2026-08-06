# Explainable, Edge-Efficient Deep Learning for Olive Leaf Disease Detection

Narrative draft of the experimental work. Numbers are quoted from
`compression/results/` and `analysis/results/`; regenerate the underlying tables
with `python3 make_report_tables.py`.

---

## 1. Data preparation: the published split leaks

The dataset ships with a train/test split, and taking it at face value inflates
every score computed on it. Two mechanisms were found.

**Duplicate photographs.** A byte-level check finds 24 images present in both
splits, which is what a naive MD5 pass reports. That figure is badly misleading.
Of the 100 train/test pairs that share a filename, only 19 are byte-identical;
the remaining 81 have different MD5 sums yet are pixel-identical at their common
800×600 resolution, with a mean absolute error of 0.0000 between 32×32
thumbnails. They are the same photograph, re-encoded. Switching detection from a
byte hash to a 16×16 greyscale thumbnail comparison raises the count to **119
duplicate pairs covering 114 training images** — nearly five times what MD5 sees.

**Burst near-duplicates.** Filenames of the form `IMG_YYYYMMDD_HHMMSS.jpg` carry
a capture timestamp, and those timestamps run in consecutive seconds: they are
burst sequences of a single leaf. 52.6% of the test images for
`olive_peacock_spot` have a training image taken within ten seconds. Neither hash
catches these, because the camera angle shifts slightly between frames, but to a
convolutional network they are images it has already seen.

The official test split was left **completely intact** so that results remain
comparable with other work on this dataset; 359 images were removed from the
training side only. The train/validation split was rebuilt to be stratified by
class and grouped by leaf identity, with bursts and duplicates merged into
groups by union-find. This replaces `ImageDataGenerator(validation_split=0.3)`,
whose behaviour is to take the last 30% of each class's *sorted filenames* —
and since the filename prefix encodes the capture device (`B-*` camera, `IMG_*`
phone, `DSC_*` DSLR), that default produced a validation set systematically
shifted away from the training distribution.

| class | original | after de-duplication | train | val | test |
|---|---|---|---|---|---|
| Healthy | 830 | 723 | 578 | 145 | 220 |
| aculus_olearius | 690 | 684 | 547 | 137 | 200 |
| olive_peacock_spot | 1200 | 954 | 763 | 191 | 260 |
| **total** | **2720** | **2361** | **1888** | **473** | **680** |

Verification: 1505 groups, none straddling the train/validation boundary; class
proportions agree across the two splits to within 0.1 percentage points.

**Evidence that this mattered.** Trained on the original split, the DenseNet121
transfer model reported validation macro-F1 0.819 against test 0.946 — a
13-point gap in the wrong direction, since a held-out test set should be no
easier than validation. After de-leaking and regrouping, the same architecture
gives validation 0.9237 against test 0.9387, a gap of 1.5 points. The anomaly
disappears, which is the clearest indication that it was leakage rather than
chance.

---

## 2. Baselines: the compact architecture is already competitive

Two backbones were fine-tuned on the cleaned split using the same staged recipe:
freeze the backbone and train the head, then unfreeze the final block at a much
lower learning rate, keeping BatchNorm in inference mode throughout. Random
seeds fix weight initialisation as well as data order, and models are saved with
`include_optimizer=False` so that reported file sizes measure the model rather
than the optimiser state.

| backbone | parameters | train time | val macro-F1 | test macro-F1 |
|---|---|---|---|---|
| MobileNetV2 | 2,261,827 | 20 min | 0.9142 | **0.9296** |
| DenseNet121 | 7,040,579 | 58 min | 0.9237 | **0.9387** |

DenseNet121 spends **3.1× the parameters and 2.9× the training time to gain 0.9
percentage points**. That result alone supports the edge-efficiency argument
before any compression is applied, and it foreshadows the compression findings:
a network that carries three times the parameters for less than a point of extra
accuracy is carrying a great deal of slack.

The two models are not merely close in aggregate; they are complementary.
MobileNetV2 is stronger on `Healthy` (recall 0.950 against 0.895) while
DenseNet121 is markedly stronger on `aculus_olearius` (0.970 against 0.885), the
minority class.

---

## 3. Emerging technique: model compression

Quantisation uses `tf.lite`, the converter built into TensorFlow 2.12. Pruning is
implemented directly against Keras weights — magnitude masks applied through
`get_weights()`/`set_weights()`, with the masks re-applied after every training
batch so the optimiser cannot revive pruned weights. No package outside the
course environment is installed at any point.

### 3.1 Quantisation: architecture decides how much precision can be given up

| variant | MobileNetV2 | | DenseNet121 | |
|---|---|---|---|---|
| | size | test macro-F1 | size | test macro-F1 |
| TFLite float32 | 8.88 MB | 0.9296 | 27.91 MB | 0.9387 |
| float16 | 4.48 MB | 0.9282 | 14.06 MB | **0.9387** |
| dynamic range int8 | 2.51 MB | **0.9341** | 7.42 MB | 0.9315 |
| full integer int8 | 2.72 MB | 0.9242 | 7.36 MB | **0.8676** |

Float16 is effectively lossless on both, and bit-identical on DenseNet121.
Dynamic-range quantisation costs nothing measurable. The divergence appears at
full integer quantisation, where MobileNetV2 gives up 0.6% of macro-F1 while
DenseNet121 gives up **7.6%** — a thirteenfold difference in damage from the same
operation. The per-class breakdown shows DenseNet121 collapsing toward the
majority class: `Healthy` recall falls from 0.895 to 0.727 and `aculus_olearius`
from 0.970 to 0.855, while `olive_peacock_spot` rises to 0.996.

The likely mechanism is architectural. DenseNet concatenates feature maps from
many layers, so a single tensor spans a wide dynamic range, and per-tensor int8
quantisation represents that range with one scale factor. MobileNetV2's linear
bottlenecks keep tensor ranges comparatively uniform and tolerate the coarser
representation.

### 3.2 Pruning: the sparsity curves are the same shape, 45 points apart

Each sparsity level prunes a fresh copy of the baseline and fine-tunes to
convergence (up to 30 epochs at 1e-4, with early stopping and learning-rate
decay).

| sparsity | MobileNetV2 | DenseNet121 |
|---|---|---|
| 0% | 0.9296 | 0.9387 |
| 20% | 0.9201 (−0.95) | — |
| 35% | 0.8892 (−4.04) | — |
| 50% | 0.8328 (−9.69) | **0.9450 (+0.63)** |
| 65% | 0.5343 (−39.53) | **0.9344 (−0.43)** |
| 80% | — | 0.8291 (−10.96) |
| 90% | — | 0.6674 (−27.13) |

At the same 65% sparsity, MobileNetV2 loses 39.5 points and DenseNet121 loses
0.43. The free ceiling — the point past which accuracy starts to decline — sits
near 20% for MobileNetV2 and near 65% for DenseNet121, a factor of 3.25.

A practical note for anyone reproducing this: **the accuracy immediately after
pruning does not predict the accuracy after recovery.** MobileNetV2 at 50%
sparsity scores 0.156 before fine-tuning and 0.833 after; at 65% it scores
higher before fine-tuning (0.223) and far worse after (0.534). Intermediate
readings are not informative, and an under-converged recovery misrepresents the
cost of compression — a first pass here used 12 epochs at 1e-5 and reported the
50% point as 0.706 rather than 0.833, understating the achievable accuracy by
12.7 points.

### 3.3 Re-reading the curves by capacity rather than by percentage

Plotting the same results against the number of surviving non-zero weights
reorganises them:

| non-zero weights | model | test macro-F1 |
|---|---|---|
| ~3.5M | DenseNet121 @ 50% | 0.9450 |
| ~2.4M | DenseNet121 @ 65% | 0.9344 |
| 2.26M | MobileNetV2, unpruned | 0.9296 |
| ~1.8M | MobileNetV2 @ 20% | 0.9201 |
| ~1.5M | MobileNetV2 @ 35% | 0.8892 |
| ~1.4M | DenseNet121 @ 80% | 0.8291 |
| ~0.7M | DenseNet121 @ 90% | 0.6674 |

The two architectures perform almost identically wherever they carry comparable
numbers of effective weights, and both degrade sharply below roughly 1.5M. The
threshold — around **2.3M non-zero weights** for this task — appears to be a
property of the problem rather than of either network. What the architecture
determines is where you start and how much slack you have to remove before
reaching that threshold. Designing a compact network and pruning a large one are
two routes to the same destination.

### 3.4 Deployment: pruning does not make anything faster

Latency was measured single-threaded on CPU with a fixed thread pool, ten warm-up
invocations, and the median of a hundred timed runs.

| DenseNet121 variant | file | gzipped | latency | test macro-F1 |
|---|---|---|---|---|
| Keras baseline | 38.15 MB | 27.06 MB | 119.6 ms | 0.9387 |
| pruned 50% | 38.21 MB | 16.53 MB | 119.1 ms | 0.9450 |
| pruned 65% | 38.23 MB | 12.92 MB | 119.2 ms | 0.9344 |
| pruned 90% | 38.23 MB | 6.14 MB | 119.5 ms | 0.6674 |
| TFLite float32 | 27.91 MB | 26.08 MB | 110.3 ms | 0.9387 |
| TFLite full integer | 7.36 MB | 5.86 MB | **57.5 ms** | 0.8676 |
| pruned 50% + dynamic range | 7.42 MB | 4.67 MB | 106.6 ms | **0.9489** |

Removing 90% of the weights leaves latency unchanged at 119.5 ms and the file
size unchanged at 38.23 MB. Unstructured pruning zeroes weights without altering
tensor shapes, so the arithmetic performed at inference is identical and the
serialised model is the same size; the benefit appears only after entropy
coding, where the gzipped model falls by 77%. Quantisation, by contrast, halves
latency. This reproduces, on our own data, the conclusion Kuzmin et al. (2023)
reach in general: for deployment, quantisation is the effective lever, and
unstructured pruning is worth having mainly where transmission or storage
dominates.

Two further latency observations are worth reporting because they cut against
intuition. First, **the smallest model is not the fastest**: on MobileNetV2,
dynamic-range quantisation produces the smallest file at 2.51 MB but runs at
18.56 ms, slower than the 8.88 MB float32 model at 11.37 ms, because int8 weights
must be dequantised at runtime while activations stay in float. Full integer
quantisation, which keeps both in int8, is both small and fastest at 9.35 ms.
Second, **most of the available speed-up comes from the runtime, not from
numerical precision**: converting the Keras model to TFLite without any
optimisation takes MobileNetV2 from 53.69 ms to 11.37 ms, a 4.7× gain with the
weights untouched; quantisation then adds a further 1.2×.

### 3.5 What to deploy

No single variant wins on every axis, which is the practical finding.

| constraint | choice | size | latency | test macro-F1 |
|---|---|---|---|---|
| smallest download | MobileNetV2, dynamic range | 2.51 MB | 18.6 ms | 0.9341 |
| fastest inference | MobileNetV2, full integer | 2.72 MB | 9.4 ms | 0.9242 |
| highest accuracy | DenseNet121, pruned 50% + dynamic range | 7.42 MB | 106.6 ms | 0.9489 |

For a phone or single-board computer performing field diagnosis, MobileNetV2 with
full integer quantisation is the defensible default: an eleven-fold reduction
against the unconverted Keras model, a 5.7× speed-up, and 0.5 percentage points
of macro-F1 given up.

---

## 4. Explanation fidelity under compression

Compression is only acceptable if the compressed model reasons about the image in
the same way. Grad-CAM cannot answer this, because a `.tflite` model exposes no
gradients. LIME can: it is model-agnostic, requiring only a function from images
to class probabilities, so the identical procedure runs against the Keras
baseline and every quantised variant. Both explanations use the same fixed SLIC
segmentation and the same random seed, so any difference in the fitted weights is
attributable to the models rather than to sampling.

| variant | mean Spearman ρ | mean top-5 Jaccard | label agreement |
|---|---|---|---|
| float16 | 0.599 | 0.833 | 100% (6/6) |
| dynamic range | 0.556 | 0.548 | 83% (5/6) |
| full integer | 0.407 | 0.627 | 100% (6/6) |

The ordering follows the accuracy loss: the variant that costs least accuracy
also perturbs the explanation least. Explanation fidelity and predictive accuracy
do not come apart here.

The two metrics should be read together. A rank correlation of 0.4–0.6 indicates
that the ordering of evidence shifts appreciably. The top-5 overlap is much
higher — under float16, four of the five most influential regions are shared on
average — which locates the instability in the low-weight regions rather than in
the evidence the prediction actually rests on.

Label agreement is not uniform, and the exception is worth stating rather than
rounding away: float16 and full-integer agreed with the baseline on all six
images, dynamic range on five. So for two of the three variants the comparison is
between models giving the same answer for possibly different reasons, which is
the question worth asking; for dynamic range, one image out of six changed answer
outright. Six images is far too small a sample to put a rate on that, and it is
reported here as the raw count for exactly that reason.

---

## 5. Does the model learn the disease, or the camera?

The dataset's filenames encode the capture device, and device correlates
alarmingly well with class in the training split.

| capture source | n (train) | dominant class | purity |
|---|---|---|---|
| bare numeric (`1.jpg` …) | 690 | `aculus_olearius` | **100.0%** |
| `B-*` camera roll | 610 | `Healthy` | **99.8%** |
| `A*` camera roll | 250 | `olive_peacock_spot` | 88.8% |
| `IMG_*` phone | 1151 | `olive_peacock_spot` | 83.6% |
| `DSC_*` DSLR | 19 | `olive_peacock_spot` | 78.9% |

Every one of the 690 training images for `aculus_olearius` has a bare numeric
filename, and no image of any other class does. A classifier given nothing but
the filename prefix reaches **91.8% accuracy on the training split**, against a
38.2% majority baseline. Any training accuracy on this dataset must be read with
that in mind. The official test split partially breaks the correlation, since its
source mix differs, and the same rule reaches only 49.9% there.

Whether the model exploits the shortcut is a separate question, and a measurable
one. Grad-CAM++ attention was integrated inside and outside a leaf mask — the HSV
rule from the project's own feature extractor, so both components agree on what
counts as leaf — over 75 randomly sampled test images.

| subset | n | attention outside leaf | background area | ratio |
|---|---|---|---|---|
| all | 75 | 10.9% | 15.5% | **0.70** |
| correct predictions | 72 | 10.8% | 15.3% | 0.71 |
| incorrect predictions | 3 | 13.0% | 19.3% | 0.67 |

A ratio below 1 means attention concentrates on the leaf more than an even spread
would produce, and errors are not explained by background reliance. **The
confound is present in the data, but on average the model is not using it.**

That average conceals a real failure mode. `a245.jpg`, a true `aculus_olearius`,
is classified as `olive_peacock_spot` with 0.979 confidence. Its prefix is `a*`,
and in training the `A*` source is 88.8% `olive_peacock_spot`; only 9 of the 200
test images for this class share that prefix, and its background is a grey-green
surface rather than the white paper used elsewhere. The model is not
background-driven in general, but it is misled on individual images whose capture
context is rare for their class — a more useful statement for a deployment
discussion than either blanket claim.

---

## 6. Glass-box comparison

Two interpretable models were fitted to nine engineered features — five colour
fractions measured inside a leaf mask, four GLCM texture descriptors. They are
evaluated under two protocols, because the protocol turns out to change the
answer.

The **common** protocol extracts the same features from exactly the images the
CNNs use, and scores on the official 680-image test split. These are the only
glass-box numbers that can be placed beside a CNN:

| model | accuracy | macro-F1 |
|---|---|---|
| **Logistic regression, 47 features** | **0.746** | **0.742** |
| Decision tree (depth 3), 9 features | 0.684 | 0.667 |
| Logistic regression, 9 features | 0.621 | 0.600 |
| Decision tree (depth 3), 47 features | 0.600 | 0.552 |

The **segmented** protocol is the setting the features were designed for: one
crop per detected leaf, held out with a split stratified and grouped by source
photograph, since one photograph can yield several crops and siblings on opposite
sides of the boundary would leak. It is reported as a sensitivity analysis, not
as a comparison against the CNNs:

| model | accuracy | macro-F1 |
|---|---|---|
| **Logistic regression, 47 features** | **0.837** | **0.831** |
| Decision tree (depth 3), 47 features | 0.693 | 0.679 |
| Logistic regression, 9 features | 0.711 | 0.692 |
| Decision tree (depth 3), 9 features | 0.624 | 0.621 |

Two things move the ranking, and both matter. On the nine base features the
protocol reverses it: the colour fractions are measured inside a leaf mask that a
tight crop makes reliable and a whole photograph does not, so logistic regression
loses on full photographs while a depth-3 tree's thresholds absorb some of the
shift. The 47-feature set then reverses it back, because logistic regression has
texture and morphology descriptors to fall back on. It also costs the tree, which
consults three features however many it is offered and picks worse ones from 47.
Any claim about *which* glass-box model is better has to name both the protocol
and the feature set. Coefficients still read directly in agronomic terms: `frac_yellow`, the
chlorosis proxy, is the largest positive weight for `olive_peacock_spot` and the
largest negative one for `Healthy`.

Both models are also explained per prediction — two correct and two incorrect
test cases each, with the decision path or the exact per-feature contribution
behind them. Under the common protocol both of the tree's explained errors turn
on the very first split, `frac_yellow <= 0.0134`, one of them missing the
threshold by 0.03 standard deviations. The errors are not diffuse; they are a
knife-edge on a single feature, which is the kind of statement only an intrinsic
explanation supports.

---

## 7. Limitations

**Three sweep points did not converge.** MobileNetV2 at 65% and DenseNet121 at
65% and 90% ran the full 30-epoch budget without triggering early stopping, so
those figures are lower bounds within that budget rather than converged results.
MobileNetV2 at 65% in particular is likely understated.

**Grad-CAM resolution.** DenseNet121's final convolutional layer produces a 7×7
grid, upsampled to 224×224 for display. The heatmaps cannot localise individual
lesions; the strongest claim the evidence supports is attention to leaf tissue.
The same coarseness blurs attention across the leaf boundary and therefore
inflates the measured share falling outside it, which makes the 0.70 ratio a
conservative estimate.

**LIME sample size.** Fidelity was measured on six images, two per class, at 500
perturbations each. The ordering across variants is consistent, but the absolute
correlations rest on a small sample.

**Single machine, CPU only.** All latency figures come from one 20-thread
i7-1280P with a single-thread pool. They are internally comparable but should not
be read as absolute numbers for any particular edge device. No GPU was available:
the course environment is CPU-only, and on the Azure student subscription every
GPU family with quota is retired while the families still offered have zero
quota.

**Structured pruning was not attempted.** Only unstructured magnitude pruning was
implemented, which is why no latency benefit appears. Structured (channel)
pruning would change tensor shapes and could deliver real speed-ups; DenseNet's
concatenation topology makes it substantially harder to implement.

---

## 8. Where these results answer the research questions

**RQ a** — pretrained CNNs against training from scratch and against a glass-box
baseline — is addressed by sections 2 and 6. Every figure below is test macro-F1
on the same official 680-image split:

| | test macro-F1 | parameters |
|---|---|---|
| DenseNet121 (ImageNet) | 0.939 | 7,040,579 |
| MobileNetV2 (ImageNet) | 0.930 | 2,261,827 |
| Scratch CNN, redesigned | 0.871 | 585,059 |
| Scratch CNN, notebook architecture | 0.810 | 5,767,139 |
| Logistic regression, engineered features | 0.742 | 47 features |
| Decision tree, engineered features | 0.667 | 9 features |

Three things follow, and only the first is the expected one.

**Transfer learning wins, but by less than the literature would suggest.** The
margin over a competently designed scratch CNN is 6 points of macro-F1, not the
collapse the original notebook implied. What ImageNet initialisation buys most
clearly here is *time*: MobileNetV2 reached 0.930 in 1,175 seconds, while the
redesigned scratch network needed 2,632 seconds to reach 0.871.

**The original notebook's 0.401 was not evidence that scratch training fails on
this task.** Reproducing that architecture unchanged on the corrected split
yields 0.810 — double the reported figure. The remaining gap to 0.871 is
architectural: that design spends 5.75M of its 5.77M parameters on a single
`Flatten` into `Dense(64)`, so a tenth of the parameter budget spent on depth,
BatchNorm and global average pooling beats it outright. Both causes have to be
named; attributing the 0.401 to either alone would be wrong.

**The glass-box gap is 20 points.** The comparison must be quoted from the common
protocol *and* from the best feature set: the nine base features reach only 0.667,
and the segmented-crop figure of 0.831 is measured on a different population. Note
also that the ranking of the two interpretable models reverses between protocols
and between feature sets (section 6), so any claim about which glass-box model is
stronger has to name both.

One caveat belongs with this table: the redesigned scratch CNN is the only model
whose validation score (0.916) overstates its test score (0.871). Validation is
carved from the training photographs and the test split is the dataset's own, so
a model whose features are learned entirely from these photographs can lean on
what they share. The two transfer models show the opposite, smaller gap.

**RQ b** — how far quantisation and pruning reduce size and latency while
preserving accuracy and explanation fidelity — is addressed by sections 3 and 4.
The short answer is that compression is cheap but architecture-dependent: what an
architecture will give up depends on where its redundancy lies, DenseNet121's in
the number of weights and MobileNetV2's in their precision. Both converge on the
same effective-capacity threshold of roughly 2.3M non-zero weights, which appears
to be set by the task. For deployment specifically, quantisation halves latency
while unstructured pruning does not improve it at all, and explanation fidelity
degrades in step with accuracy rather than independently of it.

Section 1 supports the data-exploration and preprocessing requirement, section 5
the bias investigation, and section 7 the limitations discussion.

**Suggested citations.** Kuzmin et al. (2023) for the pruning-versus-quantisation
comparison in 3.4; Han et al. (2016) for the compression pipeline in general;
Sandler et al. (2018) for MobileNetV2's inverted-residual design in the
explanation offered in 3.1; Ribeiro et al. (2016) for LIME in section 4;
Selvaraju et al. (2017) for Grad-CAM in section 5.
