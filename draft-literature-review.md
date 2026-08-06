# Literature review

Draft organised under the five headings the project plan specifies (item 6), with
the reference table item 7 asks for. Every claim below was checked against the
PDF in `ref/`; nothing is taken from a secondary summary.

Ten papers are available locally. Section 6 lists what is still missing and would
need to be sourced before submission.

---

## 1. Plant disease image classification with deep learning

The foundational demonstration is Mohanty, Hughes and Salathé (2016), who trained
AlexNet and GoogLeNet on **54,306 images spanning 14 crop species and 26
diseases** and reached **99.35% accuracy on a held-out test set**. The paper is
usually cited for that figure, but its more useful contribution for this project
is the framing: the authors position the work explicitly as a step toward
*smartphone-assisted* diagnosis, motivated by global smartphone penetration
rather than by laboratory accuracy. That framing is the direct ancestor of the
edge-efficiency question addressed here.

The paper is equally important for what it concedes. The 99.35% figure is
obtained on images captured under controlled conditions, and the authors are
explicit that performance degrades on images taken in settings different from the
training data. Every subsequent finding in this project — the capture-source
confound, the identical 800×600 curation of our own dataset, the individual
failure on an image with an unusual background — is an instance of that caveat.
It should be cited when justifying the choice of an image-classification pipeline
*and* when discussing the limits of the resulting numbers.

Foysal, Ahmed and Haque (2024) update the same pattern with a CNN over **14 plant
classes and 26 diseases at 98.14% accuracy**, integrated into a mobile
application for real-time diagnosis. It is not olive-specific and adds little
methodologically, but it is the most recent evidence that the pipeline
classification → mobile deployment is an active, practical target rather than a
speculative one, which supports the human-centred framing of the project.

## 2. Olive leaf disease detection and dataset limitations

**This is the weakest-supported section and the honest gap in the review.** None
of the ten papers held locally is olive-specific. The proposal's claims about
VGG16/VGG19 transfer learning on olive datasets, about CNN + Vision Transformer
hybrids, and about the climate-warming and yield-loss figures are currently
uncited. Two options: source those references, or soften the claims to what the
evidence supports. See section 6.

What can be said from the data rather than the literature is that the dataset
used here has three classes covering two conditions plus healthy tissue, roughly
3,400 images before de-duplication, from one partial season in 2019, with no
cultivar or geography recorded. It should not be described as supporting general
olive disease diagnosis, and the review should say so rather than leaving the
reader to infer it.

## 3. Explainability for CNN image classifiers

Selvaraju et al. (2017) introduce **Grad-CAM**, which uses the gradients of a
target class flowing into the final convolutional layer to produce a localisation
map over the input. Its practical advantage, and the reason it is used here, is
that it applies to a wide range of CNN families **without architectural change or
retraining**. The paper is candid that the map it produces is *coarse*, which is
the direct source of a limitation in this project: DenseNet121's final
convolutional layer is a 7×7 grid, so the upsampled heatmaps cannot localise
individual lesions, and the strongest claim the method supports is attention to
leaf tissue rather than to specific symptoms.

Sagar, Javed and Doermann (2023) survey leaf-based plant disease detection and
XAI together, covering traditional and deep techniques and the available
datasets. It is the appropriate anchor for the "why XAI in agriculture"
argument — more current than Mohanty and directly on topic — and it motivates
interpretability in terms of end-user trust rather than of research interest.

Amara, König-Ries and Samuel (2024) apply **Automated Concept-based Explanation
(ACE)** to plant disease classification with InceptionV3 on PlantVillage. This
matters here for two reasons. It demonstrates a class of explanation *above* the
pixel level — identifying recurring visual concepts rather than highlighting
regions — which is the natural answer to Grad-CAM's resolution limit. And
concept-based analysis is precisely the tool that would expose a background or
capture-device confound, which is the confound measured directly in this
project's bias investigation. It is the right citation for arguing that
region-level heatmaps are a floor rather than a ceiling.

For method choice, the relevant point for this project is that Grad-CAM requires
gradients. A quantised TFLite model has none, so explaining a compressed model
requires a model-agnostic method — LIME here — and that constraint should be
attributed to the method's design rather than presented as a preference.

## 4. Efficient architectures for edge deployment

Sandler et al. (2018) introduce **MobileNetV2**, built on an inverted residual
structure with shortcut connections between thin bottleneck layers, lightweight
depthwise convolutions in the expansion layer, and — the design point the paper
emphasises — the *removal of non-linearities in the narrow layers* to preserve
representational power. The authors evaluate the accuracy/operations trade-off
using multiply-adds **and actual latency**, not parameter count alone.

Both details matter for the results reported here. The linear-bottleneck design
keeps activation ranges comparatively uniform across tensors, which is the most
plausible explanation for why MobileNetV2 loses only 0.6% of macro-F1 under full
integer quantisation while DenseNet121 loses 7.6%: per-tensor int8 quantisation
represents a wide dynamic range poorly, and DenseNet's concatenated feature maps
span exactly such a range. And the paper's insistence on measuring latency rather
than multiply-adds anticipates this project's finding that the smallest model is
not the fastest.

Tan and Le (2019) propose **EfficientNet**, scaling depth, width and resolution
together by a compound coefficient, with EfficientNet-B7 reaching 84.3% ImageNet
top-1 while being **8.4× smaller and 6.1× faster at inference** than the previous
best. The paper is the standard citation for the principle that efficiency is an
architectural choice rather than a post-hoc adjustment — which is the same
conclusion this project reaches from the opposite direction, by showing that
DenseNet121 needs 3.1× the parameters of MobileNetV2 for 0.9 points of macro-F1.

EfficientNetB0 was named in the project proposal but not trained; that deviation
should be acknowledged rather than quietly dropped.

Silva and Almeida (2024) provide the most directly comparable deployment study:
InceptionV3, MobileNetV1, MobileNetV2 and VGG-16 evaluated on **resource-
constrained devices including a Raspberry Pi 4B**, using **pruning and
quantisation-aware training**, reaching inference **up to 1.48× faster on an Edge
TPU for VGG16 and up to 2.13× faster with precision reduction on an Intel NCS2
for MobileNetV1**, compared against an RTX 3090, while holding accuracy. Two
points transfer. Their speed-ups come from precision reduction, consistent with
this project's finding that quantisation halves CPU latency while unstructured
pruning does not reduce it at all. And their use of dedicated accelerators is the
reason our CPU-only numbers should be presented as internally comparable rather
than as absolute edge performance.

## 5. Compression: pruning and quantisation

Han, Mao and Dally (2016), **Deep Compression**, established the three-stage
pipeline of pruning, trained quantisation with weight sharing, and Huffman
coding, reducing storage **35× to 49× without loss of accuracy** — AlexNet from
240 MB to 6.9 MB. Pruning alone removes 9× to 13× of connections; quantisation
then cuts each remaining weight from 32 bits to 5. Critically, the paper
*retrains after each stage*, which is the same requirement this project ran into:
recovery fine-tuning after pruning is not optional, and an under-converged
recovery misrepresents the cost of the sparsity.

The paper is also the origin of the entropy-coding step that explains an
initially confusing result here. Unstructured pruning zeroes weights without
changing tensor shapes, so the serialised model does not shrink and the
arithmetic at inference is unchanged; the saving appears only after entropy
coding. In our measurements, DenseNet121 pruned to 90% sparsity keeps the same
38.23 MB file and the same 119.5 ms latency, while its gzipped size falls 77%.

Kuzmin et al. (2023) address the comparison directly, noting that prior work
offered only ad-hoc comparisons. They provide an analytical treatment of expected
pruning and quantisation error, per-layer lower bounds, and an experimental
comparison across **9 large-scale models on 4 tasks**. Their conclusion is that
**quantisation outperforms pruning in most cases**, with pruning preferable only
at very high compression ratios.

This project reproduces that conclusion on its own data and adds an architectural
qualification. Quantisation is the effective lever for deployment because it is
the only one that reduces latency. But *which* technique a given network tolerates
depends on where its redundancy sits: DenseNet121 absorbs 65% sparsity for a
0.43-point loss yet loses 7.6 points to int8 quantisation, while MobileNetV2 does
the opposite. Kuzmin et al. compare the techniques; the finding here is that the
comparison has an architecture-dependent term.

---

## 6. Reference table

| paper | task | dataset | method | XAI used | edge / compression relevance | limitations |
|---|---|---|---|---|---|---|
| Mohanty et al. (2016) | plant disease classification | PlantVillage, 54,306 images, 14 crops, 26 diseases | AlexNet, GoogLeNet transfer learning | none | motivates smartphone diagnosis | 99.35% is under controlled conditions; degrades on field images |
| Foysal et al. (2024) | plant disease classification | multi-crop, 14 classes, 26 diseases | CNN + mobile app | none | real-time mobile deployment | not olive-specific; no compression analysis |
| Selvaraju et al. (2017) | visual explanation | ImageNet, VQA, captioning | Grad-CAM | Grad-CAM (introduces it) | none | maps are coarse; needs gradients, so unusable on quantised models |
| Sagar et al. (2023) | survey | multiple leaf-disease datasets | survey of traditional + deep methods | surveys XAI methods | discusses practical deployment | survey, not new results |
| Amara et al. (2024) | plant disease classification | PlantVillage | InceptionV3 + Automated Concept-based Explanation | ACE, concept-level | none directly | PlantVillage only; concept extraction is costly |
| Sandler et al. (2018) | architecture | ImageNet, COCO, VOC | MobileNetV2, inverted residuals, linear bottlenecks | none | designed for mobile; measures real latency | general-purpose, not agricultural |
| Tan & Le (2019) | architecture | ImageNet + transfer sets | EfficientNet compound scaling | none | 8.4× smaller, 6.1× faster than prior best | requires neural architecture search |
| Silva & Almeida (2024) | leaf disease classification | own thermal image dataset | InceptionV3, MobileNetV1/V2, VGG-16 + pruning and QAT | none | Raspberry Pi 4B, Edge TPU, NCS2; 1.48–2.13× speed-ups | thermal imaging, not RGB; different modality from ours |
| Han et al. (2016) | compression | ImageNet (AlexNet, VGG) | pruning + trained quantisation + Huffman coding | none | 35–49× storage reduction | storage-focused; no latency claim for unstructured sparsity |
| Kuzmin et al. (2023) | compression comparison | 9 models, 4 tasks | analytical + empirical pruning vs quantisation | none | quantisation usually wins; pruning only at extreme ratios | no architecture-conditioned analysis |

---

## 7. Missing citations — open, needs a team member

**These are not fixed and are not being fixed by whoever wrote this section.**
They are listed here so somebody can claim one and close it.

Every claim below currently sits in `proposal.txt`, in three sentences that will
be lifted into the report's Introduction. That is what makes them urgent: they
are not buried in a draft, they are on the path into the submitted document.
Each needs either a real citation added to `ref/` and to the reference list, or
the sentence weakened to what the evidence supports.

| # | Claim | Where it is now | What closes it | Priority |
|---|---|---|---|---|
| 1 | LIME as a method | used in `compression/src/xai_fidelity.py`, cited nowhere | Add Ribeiro, Singh & Guestrin (2016), *"Why Should I Trust You?"*, KDD. Non-negotiable: we run the method and do not cite it. | **highest** |
| 2 | Olive-specific VGG16/VGG19 transfer learning "has demonstrated strong performance", augmentation "confirmed important" | `proposal.txt:9` | Find the olive-leaf transfer-learning papers this refers to, or reduce the sentence to the general plant-disease literature we do hold (Mohanty et al.) | high — it is the project's premise |
| 3 | Mediterranean warming "20% faster than the global average"; yield losses "approximately 20% or higher" | `proposal.txt:7` | An agronomic or climate source for each figure. Two separate numbers, so potentially two sources. If neither is found, delete the figures and keep the qualitative claim. | medium |
| 4 | CNN + Vision Transformer hybrids for leaf disease | `proposal.txt:9` | A citation, or cut the sentence. Nothing in this project depends on it, so cutting costs nothing. | low — cheapest to close by deletion |
| 5 | Raspberry Pi / Jetson Nano deployment of plant-disease CNNs | `proposal.txt:11` | Partially covered by Silva & Almeida (2024), but that is *thermal* imaging on a Pi 4B, not RGB. Either add an RGB source or restrict the sentence to what Silva & Almeida actually show. | medium |

Two notes for whoever picks these up. Claim 4 is the only one that can be closed
by deletion alone. Claim 1 is the only one where the absence is a defect in the
work rather than in the writing — we use the method, so it must be cited whatever
happens to the rest of the introduction.
