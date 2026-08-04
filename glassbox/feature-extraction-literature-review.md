# Feature Extraction for Leaf Disease Classification — Literature Review

**Scope.** How published work turns a leaf photograph into a small table of named,
human-readable features, and which of those features are reported to matter. Written to
support research question (a) of the proposal — the *interpretable glass-box baseline*
that the CNN and transfer-learning models are compared against.

**Caveat on sources.** ScienceDirect, MDPI and Nature block automated retrieval, so a few
entries below are summarised from abstracts, indexed metadata and publisher previews
rather than full text. Those are marked *(abstract only)*. Everything else was read from
the open-access full text.

---

## 1. The pipeline everyone uses

Across three decades of this literature the classical pipeline is remarkably stable
([Feature engineering review, *Artificial Intelligence in Agriculture* 2024](https://www.sciencedirect.com/science/article/pii/S2772375524000856);
[Detection of Plant Disease Using ML and DL, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/)):

```
acquire → preprocess (denoise, resize, colour-space convert)
        → segment (leaf from background, then lesion from leaf)
        → describe (colour + texture + shape features)
        → select (drop redundant features)
        → classify (SVM / RF / kNN / DT / boosting)
```

The 2025 review is explicit that the descriptor stage is where the accuracy is won or
lost: performance "largely depends on the quality of extracted features, which requires
domain expertise." It also lists the standard criticisms of the approach — sensitivity to
lighting and noise, poor transfer to field conditions, and manual effort — which are worth
quoting in your write-up as the motivation for *also* training the CNN.

---

## 2. Feature families, with the parameters papers actually report

### 2.1 Colour

The dominant symptom signal for foliar disease. Papers consistently convert out of RGB
first, because HSV/HSI and CIE L\*a\*b\* separate chroma from illumination, so a shadow does
not read as a lesion.

| Descriptor | Typical parameterisation | Source |
|---|---|---|
| Colour moments | mean, std, skewness, kurtosis per channel (12 features from 3 channels) | [2025 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/) |
| Colour moments (compact) | 6 colour features alongside 22 texture features | [Leaf Image based Plant Disease Identification using Colour and Texture Features](https://arxiv.org/abs/2102.04515) |
| Colour histograms | per-channel bins in RGB, HSV and Lab; targets chlorosis (yellowing) and necrosis (browning) | [2025 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/) |
| Colour correlogram (CC) | quantised colour bins × spatial distance — probability of a colour pair at distance *d* | [Multi-level handcrafted features for rice disease, 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666784/) |
| Colour co-occurrence matrix (CCM) | GLCM computed on HSI channels — colour *and* spatial layout in one descriptor | [2025 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/) |

Practical note: hue is circular, so a plain `mean(H)` is wrong near the red wrap-around.
Papers rarely mention this; a circular mean/variance weighted by saturation is the correct
form and is what the code in [glassbox/olive_glassbox/features.py](glassbox/olive_glassbox/features.py) already does.

### 2.2 Texture

The most-reported family, and the one that separates *spot* diseases.

- **GLCM / Haralick.** Near-universal. Reported configurations range from a compact set
  (contrast, correlation, energy, homogeneity) to the full 13-feature Haralick bank —
  ASM, contrast, correlation, variance, inverse difference moment, sum average, sum
  variance, sum entropy, entropy, difference variance, difference entropy, and two
  information measures of correlation ([tomato handcrafted+deep, *Sci Rep* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11538303/)).
  One widely-cited configuration uses 8-bit grey × 4 orientations × 2 values = 64 texture
  values; another computes 22 GLCM features on greyscale because colour is already
  encoded by the colour moments ([arXiv:2102.04515](https://arxiv.org/abs/2102.04515)).
  Distances of 1–3 px and angles of 0°/45°/90°/135° (averaged for rotation invariance) are
  the norm.
- **LBP.** Standard 3×3 / P=8, R=1 patch giving a 256-bin histogram, or the *uniform*
  variant giving 10 bins. Praised for illumination invariance and low cost, criticised for
  noise sensitivity. **Multi-channel LBP (MCLBP)** runs LBP independently on R, G and B and
  concatenates — it beat single-channel LBP on three rice datasets ([PMC11666784](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666784/)).
- **Gabor filter banks.** The historical entry point (Kulkarni & Patil, Gabor + ANN, ~91%),
  still used for orientation/scale-selective texture ([2024 feature-engineering review](https://www.sciencedirect.com/science/article/pii/S2772375524000856), *abstract only*).
- **Wavelets (DWT).** Haar, three-level decomposition into LL/LH/HL/HH sub-bands. In the
  olive-specific paper below, the third-level approximation (LL3, 28×28) was used as the
  *input* to CNN feature extractors rather than as a descriptor itself
  ([Olive Leaf Disease Detection via Wavelet Transform, *CMC* 2024](https://www.techscience.com/cmc/v78n3/55926/html)).
- **First- to fourth-order statistical moments** of the intensity distribution, plus
  entropy and gradient energy — used in the one genuinely olive-specific classical study
  ([Olive Spot Disease Detection using Analysis of Leaf Image Textures, *Procedia CS* 2020](https://www.sciencedirect.com/science/article/pii/S1877050920307511), *abstract only*).

### 2.3 Shape

Two distinct levels, and the literature routinely conflates them. Keep them separate.

**Leaf-level shape** — area, perimeter, circularity, eccentricity, solidity, extent,
aspect ratio, and **Hu moment invariants**. Relevant here because *Aculus olearius*
deforms and curls the lamina, so leaf outline irregularity is a real signal, not a nuisance.

**Lesion-level shape** — the more informative level for spot diseases, and the most
under-used. The reference treatment is
[Schwanck & Del Ponte, *Plant Pathology* 2016](https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/ppa.12526),
which measures per-lesion **area, circularity, eccentricity, length (longest axis), width
(shortest axis), elongation (length/width), roundness, solidity, perimeter, diameter**, and
crucially **location** — distance to leaf edge and to the midrib — then analyses the
spatial pattern of lesions across the lamina. Severity is the standard ratio
*S = A_diseased / A_leaf*.

**The ABCD rule**, borrowed from dermatology, is an interesting packaging of the same idea
([*Sci Rep* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11538303/)): **A**symmetry
(lesion split across perpendicular axes), **B**order (perimeter split into 8 sectors,
scored for abrupt transitions), **C**olour (count of distinct hues present from a
white→brown→black→red palette), **D**iameter (from central moments μ₂₀, μ₀₂, μ₁₁). That
gave 43 handcrafted features total.

### 2.4 Keypoint descriptors

SIFT, SURF and HOG appear regularly, usually via bag-of-visual-words. HOG integrated with
ML classifiers improved accuracy by 1.69 pp in one study. The consensus across recent
surveys is that these underperform CNN features on leaf disease and are no longer the
first choice.

### 2.5 "Deep features + classical classifier" — the modern default

Worth knowing because it is what most current olive papers actually mean by feature
extraction: a frozen ImageNet CNN produces a 512–2048-d embedding, which is then fed to
SVM/RF/LightGBM. **This is not a glass-box method** — the features have no names and no
agronomic meaning — but it is the standard high-accuracy baseline, and it is cheap for you
to run since you already have DenseNet121 wired up in the notebook.

---

## 3. Segmentation — how the leaf and the lesion get isolated

| Method | Notes from the literature |
|---|---|
| Otsu / histogram thresholding | Simple, maximises between-class variance; ignores spatial structure |
| **k-means clustering** (usually k=3) | Most-reported; 94.0% classification accuracy downstream in one texture study. In the olive spot paper, k-means gave **higher accuracy than histogram thresholding** for isolating the infected area |
| HSV/Lab thresholding | Direct chlorosis/necrosis targeting via hue and a\*/b\* ranges |
| Watershed | Efficient but over-segments |
| Canny / edge-based | 94% on corn; noise-sensitive |
| Region growing | Noise-immune, computationally heavy |

For **this** dataset the leaf-vs-background step is easy (uniform pale studio background),
so the effort belongs in the second step: lesion-vs-healthy-lamina.

---

## 4. Feature selection

Consistently reported as necessary once you pass ~50 features:

- **Filter methods** — mutual information, chi-square, F-score, ReliefF.
- **Mutual-information thresholding** — keep features whose MI exceeds the mean MI
  ([*Sci Rep* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11538303/)).
- **Wrapper** — recursive feature elimination (RFE), used with RF/XGBoost on olive
  hyperspectral data, roc-auc 0.977 (RF) and 0.955 (XGB)
  ([*Remote Sensing* 2023](https://doi.org/10.3390/rs15245683)).
- **Composite selection** — fusing several filter rankings. The most recent olive paper
  ([*Agronomy* 2026](https://doi.org/10.3390/agronomy16111057)) combines MI, chi-square,
  F-score and five custom composites, keeping 32/64/128 features, and reaches 0.988
  accuracy / 0.976 MCC with 128 features — matching the full-dimensional vector.
- **Embedded / evolutionary** — genetic algorithms, incl. GA–Bayesian optimisation fusion.

⚠️ Every one of these must be fitted **inside** the cross-validation fold, on training data
only. Selecting features on the full dataset before splitting is the single most common
leak in this literature and inflates reported accuracy.

---

## 5. Classifiers paired with handcrafted features

SVM dominates (one-vs-one or one-vs-all, linear or RBF). Reported figures span a wide
range because datasets differ enormously in difficulty:

| Classifier | Reported accuracy | Context |
|---|---|---|
| SVM (one-vs-one), 6 colour + 22 GLCM | **98.79%** ±0.57 (10-fold) | public benchmark; **82.47%** on the authors' own field images ([arXiv:2102.04515](https://arxiv.org/abs/2102.04515)) |
| SVM linear, MCLBP ⊕ colour correlogram | 99.53% / 99.4% / 99.14% | three rice datasets ([PMC11666784](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666784/)) |
| XGBoost / SVM, 12 colour + texture | 86.58% / 81.67% | rice, 3 diseases |
| Decision tree | 97.92% (rice), 95.26% (multi-crop) | controlled lab datasets |
| Random forest | 91.47% (rice), 70.14% (papaya) | ([2025 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/)) |
| kNN | 99.96% (PlantVillage), 91.66% (rice) | lab conditions |

The gap between the same method's lab and field numbers (98.79% → 82.47%) is the single
most useful fact in this table. Quote it when you discuss generalisation.

---

## 6. Olive-specific work — and the gap you can fill

Several of these use **your exact dataset**: 3,400 images from Denizli, Turkey
(1,050 healthy / 1,460 peacock spot / 890 *Aculus olearius*), which matches your
2,720-image train split (830/690/1,200) plus the 680-image test split.

| Study | Features | Classifier | Accuracy |
|---|---|---|---|
| [Olive Spot Disease Detection via Leaf Image Textures, *Procedia CS* 2020](https://www.sciencedirect.com/science/article/pii/S1877050920307511) | GLCM energy, homogeneity, entropy + 1st–4th order moments; k-means vs histogram thresholding for lesion isolation | correlation analysis | correlates infection area with texture; k-means better *(abstract only)* |
| [ViT + CNN hybrid, *Comput Intell Neurosci* 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9357740/) | none — end-to-end deep | ViT+VGG-16 | 96% multiclass (VGG-16 alone 85%, AlexNet 84%) |
| [Wavelet + feature fusion, *CMC* 2024](https://www.techscience.com/cmc/v78n3/55926/html) | Haar DWT (3-level, LL3 28×28) → EfficientNet-B7 + DenseNet-201 + ResNet-152-V2, avg-pooled to 512-d, fused | dense fusion head | 99.72% |
| [Hybrid ML/DL, Springer 2024](https://link.springer.com/chapter/10.1007/978-3-031-77426-3_11) | **LBP** + adapted ResNet50 | SGD / SVM / RF, tuned | 99.11% (RF 0.92, SVM 0.88 in the per-scenario averages) |
| [Composite feature selection + ensemble, *Agronomy* 2026](https://doi.org/10.3390/agronomy16111057) | ResNet101 + MobileNet deep features, MI/χ²/F-score composite selection to 128 | kNN / SVM / **LightGBM**, voting ensemble | 0.988 (0.9916 ensemble) |
| [Adaptive genetic algorithm, *WCMC* 2022](https://www.hindawi.com/journals/wcmc/2022/8531213/) | end-to-end deep, GA-optimised architecture | CNN | — |
| [AI-enhanced MetaFormer, *Artif Intell Rev* 2025](https://link.springer.com/article/10.1007/s10462-025-11131-y) | end-to-end deep | MetaFormer | — |

**The gap.** With the partial exception of the 2020 *Procedia* texture study, no published
olive-leaf work builds a fully named, agronomically-meaningful feature table and reports
what the classifier actually keyed on. Every high-accuracy olive result is a deep
embedding — unnameable by construction, which is exactly the explainability deficit the
proposal identifies. A glass-box feature model on this dataset is therefore a genuine,
defensible contribution, *provided you report it honestly against the ~96–99% deep
baselines rather than trying to beat them*.

---

## 7. Recommended feature set for the three olive classes

Grounded in the symptom biology *and* in a visual audit of samples from all three classes
in this repo's `train/` folders. What that audit showed:

- Single leaf per image, arbitrary rotation, on plain paper — but the paper's colour cast
  varies between shots (white, grey-green, lilac) and camera distance varies. So:
  white-balance off the background before any colour feature, and prefer dimensionless
  ratios over pixel counts.
- Peacock spot appears in **two** forms — whole-leaf chlorosis with dark rings, and green
  leaves with discrete yellow haloes around dark stippled centres. The concentric "peacock
  eye" ring is the invariant across both.
- *Aculus olearius* leaves stay **green**. The signal is outline deformation/crinkling and
  silvery matte patches, not colour.

Consequence: Healthy vs *Aculus* is the hard boundary and it is a shape + texture problem.
Peacock spot is nearly separable on colour alone, so expect early headline accuracy to
flatter the model — always report per-class metrics.

**Olive peacock spot** (*Venturia oleaginea* / *Spilocaea oleagina*) — dark, roughly
circular sooty lesions, often with a chlorotic yellow halo, mostly on the upper surface:

- lesion count, lesion count per unit leaf area
- lesion area fraction (= severity *S*), mean and max lesion area
- mean lesion circularity, eccentricity, solidity → circular lesions are the signature
- chlorotic (yellow-halo) area fraction and count, halo-to-lesion area ratio
- GLCM contrast and dissimilarity ↑, homogeneity and ASM ↓ (spots break up the lamina)
- dark-pixel fraction in L\* and low-L\* percentiles
- mean lesion-centroid distance to the leaf margin, normalised by leaf half-width — peacock
  spots sit on the lamina, not the edge
- **`les_ring_score`** — oscillation of the radial intensity profile outward from each
  lesion centroid (variance of its second derivative, or dominant FFT peak). No paper found
  in this review computes it, yet concentric rings *are* the definition of "peacock eye";
  worth proposing as a novel descriptor

***Aculus olearius*** (gall mite) — no discrete spots; diffuse silvering, bronzing,
deformation and curling:

- colour moments in HSV/Lab: mean and **variance** of hue, a\*, b\*; skewness of L\*
- circular hue mean/variance weighted by saturation
- chroma percentiles, gloss/specular fraction (silvering raises specularity)
- leaf-level shape: **solidity** (area ÷ convex-hull area — the single best deformation
  measure), **perimeter ÷ convex-hull perimeter** (>1 when the margin is crinkled),
  **radial-distance std** (std ÷ mean of centroid→boundary distance) and boundary-curvature
  std for waviness, plus eccentricity, extent and aspect ratio
- silvering fraction: connected patches of high L\* and low chroma
- LBP uniform histogram (fine-grain surface roughening) and gradient mean/std
- grey-level entropy over the lamina

**Healthy** — uniform saturated green: high GLCM homogeneity, low entropy, low lesion
count, tight L\* range.

**Deliberately exclude** background-region features from the model. They are useful as a
*diagnostic* (if background colour predicts class, your images have an acquisition
confound), but including them lets the classifier cheat, and you would then be reporting a
photographic artefact as a biological finding.

Expect roughly 55–70 features before selection, then MI or RFE inside the CV fold down to
~20–30. That is a table you can print, plot, and explain to an agronomist — which is the
whole point of the glass-box arm.

---

## 8. Note on existing code in this repo

[glassbox/olive_glassbox/features.py](glassbox/olive_glassbox/features.py) already
implements most of the above — leaf segmentation, leaf-level shape (`leaf_eccentricity`,
`leaf_solidity`, `leaf_extent`, `leaf_aspect_ratio`), colour moments with circular hue,
GLCM (32 grey levels, distances 1 and 3, four angles averaged, six properties), uniform
LBP (P=8, R=1, 10 bins), grey-level entropy, Sobel gradient energy, and lesion morphology
(`les_count`, `les_area_frac`, `les_mean_circularity`, `les_mean_eccentricity`,
`chlor_area_frac`). The parameters match what the literature reports, so extending that
module is a much shorter path than starting a new extractor.

Gaps relative to this review, if you want to go further: multi-channel LBP, a Gabor bank,
Hu moments, colour correlogram, and per-lesion spatial statistics (distance to leaf edge
and midrib) from Schwanck & Del Ponte.

---

## Sources

- [Leaf Image based Plant Disease Identification using Colour and Texture Features (arXiv:2102.04515)](https://arxiv.org/abs/2102.04515)
- [Machine Learning for Leaf Disease Classification: Data, Techniques and Applications (arXiv:2310.12509)](https://arxiv.org/abs/2310.12509)
- [Feature engineering to identify plant diseases: a comprehensive review, *Artificial Intelligence in Agriculture* 2024](https://www.sciencedirect.com/science/article/pii/S2772375524000856)
- [A Review on the Detection of Plant Disease Using ML and DL Approaches, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565507/)
- [An enhanced classification of rice plant diseases based on multi-level handcrafted feature extraction, 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666784/)
- [Synergistic use of handcrafted and deep learning features for tomato leaf disease classification, *Sci Rep* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11538303/)
- [Measuring lesion attributes and analysing their spatial patterns at the leaf scale, *Plant Pathology* 2016](https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/ppa.12526)
- [Olive Spot Disease Detection and Classification using Analysis of Leaf Image Textures, *Procedia CS* 2020](https://www.sciencedirect.com/science/article/pii/S1877050920307511)
- [Olive Disease Classification Based on Vision Transformer and CNN Models, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9357740/)
- [Olive Leaf Disease Detection via Wavelet Transform and Feature Fusion of Pre-Trained DL Models, *CMC* 2024](https://www.techscience.com/cmc/v78n3/55926/html)
- [Optimizing Olive Disease Classification Through Hybrid ML and DL Techniques, Springer 2024](https://link.springer.com/chapter/10.1007/978-3-031-77426-3_11)
- [Efficient Olive Leaf Disease Detection Using Composite Feature Selection and Ensemble Learning, *Agronomy* 2026](https://doi.org/10.3390/agronomy16111057)
- [Application of ML for Disease Detection Tasks in Olive Trees Using Hyperspectral Data, *Remote Sensing* 2023](https://doi.org/10.3390/rs15245683)
- [Optimal Deep Learning Model for Olive Disease Diagnosis Based on an Adaptive Genetic Algorithm, *WCMC* 2022](https://www.hindawi.com/journals/wcmc/2022/8531213/)
- [Efficient and autonomous detection of olive leaf diseases using AI-enhanced MetaFormer, *Artif Intell Rev* 2025](https://link.springer.com/article/10.1007/s10462-025-11131-y)
- [Olive Leaf Image Dataset (Kaggle)](https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset)
