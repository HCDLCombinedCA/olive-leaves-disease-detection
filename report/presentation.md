# Presentation — 9 slides

Draft deck. Slide text is written to be **read off the screen in seconds**; the
argument goes in `SAY:`, not on the slide. Every number is from `RESULTS.md`.

Rules this deck follows, and that edits should keep:

- Nothing from the proposal's uncited claims (climate/yield figures, olive VGG
  studies, CNN-ViT hybrids, Pi/Jetson deployment). See `draft-literature-review.md` §7.
- Every model comparison uses the **common protocol** only.
- The system is a **triage / decision-support** aid, never "a diagnostic system".
- `[TEAM]` marks what only the team can fill.
- No new experiments. Every figure referenced already exists in the repository.

Format: 5 min per student then Q&A. With 9 slides, budget roughly one slide per
40 seconds of a 5-minute segment and leave slides 5–7 the most airtime — they
carry the marks.

---

## 1 — Title

**Explainable, Edge-Efficient Olive Leaf Disease Classification**

`[TEAM]` Names and student numbers, one line each
`[TEAM]` Module / date

`SAY:` One sentence framing: three-class olive leaf disease classification,
with the two questions being how far transfer learning actually helps, and how
far the model compresses before it stops explaining itself.

---

## 2 — Problem, dataset, research questions

- Olive peacock spot vs *Aculus olearius* vs healthy — **different treatments**, so
  telling the two diseases apart is what matters
- Olive Leaf Image Dataset (Kaggle, **CC0**) — 2,720 train / 680 test, 3 classes
- **RQ a** — ImageNet transfer vs from scratch vs interpretable glass-box models
- **RQ b** — how far can it be quantised and pruned before accuracy *or*
  explanation degrades

`SAY:` Motivate agronomically but **qualitatively** — expert diagnosis is slow
and scarce, and the hardware available in the field is constrained. Do not quote
the proposal's 20% climate/yield figures: they are currently uncited.

---

## 3 — The published split leaks

| | |
|---|---|
| Duplicate pairs across train/test | **119** |
| …found by MD5 alone | **24** |
| Train images removed (duplicate / burst) | 114 / 264 = **359** |
| Official test split | **untouched** |

Before correction: val 0.819 vs test **0.946**
After correction: val 0.914 vs test **0.930**

`SAY:` This is the slide that shows the work is trustworthy. The inversion —
scoring higher on data never validated against — is the signature of
contamination. Perceptual hashing on 16×16 thumbnails found the 95 pairs that
byte-identical hashing missed: same photograph, re-encoded. The official test
split was deliberately left intact so results stay comparable with other work.

---

## 4 — Segmentation, and what it silently produced

`FIGURE:` a leaf crop beside one `*_full.png` fallback (pick any from
`olive_leaf_dataset/segmented/train/`)

- Model chosen from an 11-configuration benchmark on CVPPP A1 → fine-tuned
  **YOLO11-seg** (SBD 0.847, 0.015 s/img)
- Applied to olive photos: **3,003 files = 2,677 detected leaves + 326 fallbacks**
- **≈11%** of the crops feeding the glass-box features are whole photographs,
  background included

`SAY:` Note carefully: **11%**, that is 326/3,003 — the share of *outputs* that
are fallbacks. A different 12% figure (326/2,720) is the share of *photographs*
where the segmenter found nothing. Do not mix them on stage. This matters because
capture background is confounded with class (slide 8), so unsegmented crops are a
route by which background can drive an interpretable model's prediction.

---

## 5 — RQ a: one test set, six models

**All rows: official 680-image test split**

| Model | Params | macro-F1 |
|---|---|---|
| DenseNet121 (ImageNet) | 7.0M | **0.939** |
| MobileNetV2 (ImageNet) | 2.3M | 0.930 |
| Scratch CNN, redesigned | 0.59M | 0.871 |
| Scratch CNN, original architecture | 5.8M | 0.810 |
| Logistic regression (47 features) | — | 0.742 |
| Decision tree (9 features) | — | 0.667 |

`SAY:` Three points, in this order.
(1) Transfer wins by **six points**, not by a collapse — and what it mainly buys
is time: 1,175 s vs 2,632 s.
(2) The 0.401 previously reported for a scratch CNN was **leakage and
architecture together** — the same architecture on the corrected split gives
0.810, and the remaining gap is that it spends 5.75M of 5.77M parameters on one
`Flatten` into `Dense(64)`.
(3) Interpretability costs **20 points** — 0.939 against 0.742. Two things must be
named when quoting it: the protocol (segmented crops give 0.831, a different test
set) and the feature set (the original nine features reach only 0.667). The
47-feature set comes from the `pooja-glassbox-1` work, re-evaluated under a
grouped split.

---

## 6 — Explaining correct *and* incorrect predictions

`FIGURE:` `analysis/results/gradcam_olive_peacock_spot.png` (2 correct, 2 wrong)
`FIGURE:` `analysis/results/glassbox_common_protocol_decision_tree_examples.png`

- **Grad-CAM++** on DenseNet121 — post-hoc, needs gradients
- **LIME** — model-agnostic, so it also runs on quantised TFLite graphs
- **Glass-box** — the explanation *is* the model: exact
  `coefficient × standardised value` for LR, the decision path for the tree
- Both tree errors turn on the **first split** (`frac_yellow ≤ 0.0134`); one
  misses by **0.03 SD**

`SAY:` The XAI debugging result is worth 20 seconds: the project's earlier
saliency maps were faithfully explaining the *broken* model — 0.401 macro-F1,
0.5% recall on *Aculus* — because the wrong checkpoint had been saved. Finding
that is what XAI bought us before any interpretation.

---

## 7 — RQ b: compression

`FIGURE:` `compression/results/compression_tradeoff.png`
`FIGURE:` `compression/results/pruning_sweep.png`

MobileNetV2:

| Variant | MB | ms | macro-F1 |
|---|---|---|---|
| Baseline | 13.2 | 53.6 | 0.930 |
| **Full integer** | **2.7** | **9.5** | 0.924 |
| Pruned 50% | 13.2 | 52.1 | 0.833 |

- Quantisation: **4.9× smaller, 5.6× faster, −0.005 macro-F1**
- Unstructured pruning: **no size or latency change** — saving shows only after gzip
- LIME fidelity tracks accuracy: label agreement float16 **6/6**, full-integer
  **6/6**, dynamic range **5/6 (83%)**

`SAY:` State the 5/6 explicitly — it is 83%, not 100%, and six images cannot
support a percentage anyway, which is why we give the count. Architecture decides
what pruning costs: DenseNet121 tolerates 50% sparsity with no loss, MobileNetV2
loses 0.097, because a network already designed for efficiency has less
redundancy to surrender.

---

## 8 — Bias, drift, and what we do *not* claim

- Filename prefix alone predicts class: **91.8%** train vs 38.2% majority
  baseline — capture source is confounded with label
- But the model is **not** exploiting it: Grad-CAM++ attention outside the leaf
  **10.9%** of a **15.5%** background → ratio **0.70**
- **Drift** = capture conditions changing, not the disease. Monitor input
  statistics, prediction confidence, and that 0.70 ratio
- Confidence is **not calibrated** — errors at 0.979 and 0.991 → abstain, never
  show raw softmax
- Positioned as **triage / decision-support**, not diagnosis

`SAY:` The framing is a conclusion from measurement, not modesty: the label set
is closed at three classes, and the errors are confident ones *between diseases*.
A triage tool that degrades wastes an agronomist's time, which is recoverable.

---

## 9 — Conclusions

1. **Auditing the data was the highest-value work.** 359 leaked images removed;
   a previously reported 0.401 became 0.810 once leakage and architecture were
   separated.
2. **Transfer learning's advantage is real but modest** — 0.939 vs 0.871 — and
   is mostly training time.
3. **Quantisation is close to free; unstructured pruning buys nothing on CPU**,
   and explanation fidelity degrades with accuracy rather than independently.

`[TEAM]` Individual contributions — one line per member on screen, detail in the
report appendix

`SAY:` Close on the limitation you most expect to be asked about: LIME rests on
six images, Grad-CAM++ at 7×7 cannot localise individual lesions, and the
capture-source confound bounds what any accuracy figure on this dataset means.

---

## Likely Q&A

| Question | Answer |
|---|---|
| Why is the glass-box comparison fair now? | Same features, same manifests, same official 680 test images. The earlier 0.692 was on crops derived from *training* photographs. |
| Why does the tree get *worse* with more features? | Depth 3 means it uses three features whatever it is offered. With 47 candidates it picks worse ones than it did from 9. Logistic regression uses all of them, so it gains. |
| Why not use TFMOT for pruning? | Not in the course image. Pruning is written directly against Keras weights, which also makes the masking explicit. |
| Why LIME and not Grad-CAM for the compressed models? | A `.tflite` graph exposes no gradients. LIME needs only inputs→probabilities, so the identical procedure runs on every variant. |
| Why did the redesigned scratch CNN score *lower* on test than validation? | Validation is carved from the training photographs; the test split is the dataset's own. A model whose features come entirely from these photographs can lean on what they share. |
| Is 6 images enough for the fidelity claim? | No, and we say so. It is reported as raw counts, and it is a limitation, not a result. |
