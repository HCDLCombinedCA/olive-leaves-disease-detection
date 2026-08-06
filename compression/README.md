# Model compression (ETech implementation #2)

Post-training quantisation and magnitude pruning for the olive leaf disease
classifier, with the size / latency / accuracy trade-off measured on CPU.

This addresses RQ b: *to what extent can quantisation and pruning reduce model
size and inference latency while preserving accuracy and explanation fidelity?*

## Scope: course toolset

Everything runs on `kquille/hcaim_gpu_tudublin_course:latest` **unmodified** —
`run.sh` installs nothing. Where each technique comes from:

| Need | Tool used | Source |
|---|---|---|
| Transfer learning, frozen backbone | `keras.applications`, `include_top=False` | Week 4 — `Transfer_Learning-No Batch Processing.ipynb` |
| Callbacks | `EarlyStopping`, `ReduceLROnPlateau` | Week 3 — `Regularization_Teachniques.ipynb` |
| Data loading | `ImageDataGenerator`, `flow_from_*` | Weeks 4, 7 |
| Model export | SavedModel, `include_optimizer=False` | Week 7 — `productionModel.ipynb` |
| Pruning | Keras `get_weights()` / `set_weights()` + `fit()` | Weeks 1, 2, 4 |
| Explanations | `lime` | Week 8 — `Lime and Shap.ipynb` |
| Quantisation | `tf.lite.TFLiteConverter`, `tf.lite.Interpreter` | no lab; core TensorFlow 2.12 API |

Quantisation is the only item with no lab behind it — a keyword search across all
27 lab notebooks returns no matches for tflite, quantisation, pruning or
sparsity. It is core TensorFlow rather than an added package, so the environment
stays exactly as delivered.

`tensorflow_model_optimization` (TFMOT) is **not** used: it is absent from the
course image, so pruning is written directly against Keras weights instead. That
also keeps the masking explicit rather than hidden behind a wrapper.

The course image is CPU-only (`tf.config.list_physical_devices('GPU')` returns
`[]`). No GPU was available locally or on the Azure student subscription, where
every GPU family with quota is retired and the families still offered have zero
quota. This is not a constraint for the results: latency must be measured on CPU
anyway to represent an edge deployment target.

## Working convention

**Do not edit teammates' sources in place.** Anything needed from
`Olive_Leaf_CNN_Model_Notebook_withSaliency.ipynb`, `glassbox/`, or
`leaf_segmenter/` is copied into `src/` and adapted here, leaving the originals
untouched. Copied code carries a header naming its origin.

Comments and documentation are written in English throughout.

## Layout

```
run.sh                  run any command inside the course image
src/prepare_data.py     extract, de-leak, and split the dataset
data/prepared/          generated dataset + manifests   (gitignored)
artifacts/              trained and compressed models   (gitignored)
results/                CSVs and figures for the report (tracked)
```

## Usage

```bash
./run.sh python src/prepare_data.py      # step 1: dataset
```

## Step 1 — dataset preparation (done)

The published dataset leaks training images into the test split. Two mechanisms,
both of which inflate test scores:

**Duplicate images.** MD5 alone under-reports this badly. Of the 100 train/test
pairs sharing a filename, only 19 are byte-identical; the other 81 have different
MD5s but are pixel-identical at 800×600 (mean absolute error 0.0000 on 32×32
thumbnails) — the same photograph, re-encoded. Detection therefore uses a 16×16
greyscale thumbnail comparison rather than a byte hash, which finds **119
duplicate pairs covering 114 training images**.

**Burst near-duplicates.** `IMG_YYYYMMDD_HHMMSS.jpg` timestamps run in
consecutive seconds — burst shots of one leaf. **264 training images** share a
burst with a test image.

Handling:

* The official test split is left **completely intact** (680 images), so results
  stay comparable with other work on this dataset.
* 359 images are removed from the training side only.
* The train/val split is **stratified by class and grouped by leaf identity**
  (bursts and duplicates merged via union-find), replacing
  `ImageDataGenerator(validation_split=0.3)`. That default takes the last 30% of
  each class's *sorted filenames*, and since the filename prefix encodes the
  capture device (`B-*` camera, `IMG_*` phone, `DSC_*` DSLR) it produced a
  validation set systematically shifted away from the training distribution.

Result:

| class | original | after de-dup | train | val | test |
|---|---|---|---|---|---|
| Healthy | 830 | 723 | 578 | 145 | 220 |
| aculus_olearius | 690 | 684 | 547 | 137 | 200 |
| olive_peacock_spot | 1200 | 954 | 763 | 191 | 260 |
| **total** | **2720** | **2361** | **1888** | **473** | **680** |

Verified: 1505 groups, **0** straddling the train/val boundary; class shares
match across the two splits to within 0.1 pp.

`--keep-leaks` reproduces the original leaky split, so the report can quantify
how much the leakage was worth in test score.

## Reading the pruning rows in the results table

Two families of pruned variants appear, produced by different recovery protocols.
They are **not** interchangeable, and the difference between them is itself a
result worth reporting.

| prefix | script | recovery protocol |
|---|---|---|
| `*_pruned50` | `src/prune.py` | 12 epochs, LR 1e-5 |
| `*_sweep{NN}` | `src/prune_sweep.py` | up to 30 epochs, LR 1e-4, early stopping + LR decay |

The first pass used 1e-5 because that is the learning rate the baseline's second
stage used — appropriate for gently adapting pretrained features, far too low for
recovering from having half the weights zeroed. Its validation curve was still
climbing at the final epoch and never plateaued, so the resulting score measures
*twelve epochs of recovery*, not the cost of the sparsity.

The sweep re-runs the same sparsity levels at 1e-4 with early stopping, so each
point stops when it has actually converged. Where both exist, **the `sweep` row is
the one to quote**; the `pruned50` row is kept as a worked example of how an
under-converged recovery misrepresents a compression result.

This is the same methodological error this project flagged in the team's original
architecture search — conclusions drawn from runs that had not converged — and it
was reproduced here in the first implementation before the sweep exposed it.

## Results

All steps are complete for both backbones. `./pipeline.sh <backbone>` reproduces
everything from a clean checkout; `python3 ../make_report_tables.py -o ../RESULTS.md`
regenerates the combined tables. `../draft.md` is the narrative write-up.

Headline findings:

* **The compact backbone is already competitive.** DenseNet121 spends 3.1× the
  parameters of MobileNetV2 for 0.9 points of test macro-F1 (0.9387 vs 0.9296).
* **Quantisation cost depends on the architecture.** Full integer quantisation
  costs MobileNetV2 0.6% of macro-F1 and DenseNet121 7.6% — DenseNet's
  concatenated feature maps span a wide dynamic range that a single per-tensor
  scale represents poorly.
* **So does pruning, in the opposite direction.** DenseNet121 tolerates 65%
  sparsity for a 0.43-point loss; MobileNetV2 loses 39.5 points at the same level.
  Its free ceiling is near 20%.
* **Both converge on the same effective capacity.** Re-plotted against surviving
  non-zero weights, the two architectures perform alike wherever they carry
  comparable numbers, and both fall away below roughly 1.5M. The threshold —
  around 2.3M — looks like a property of the task, not of either network.
* **Unstructured pruning buys nothing at inference time.** Removing 90% of
  DenseNet121's weights leaves latency at 119.5 ms against a 119.6 ms baseline and
  the file size unchanged; only the gzipped size falls (−77%). Quantisation halves
  latency. This reproduces Kuzmin et al. (2023) on our own data.
* **The smallest model is not the fastest.** Dynamic-range quantisation yields the
  smallest MobileNetV2 file (2.51 MB) but runs slower than float32, because int8
  weights are dequantised at runtime. Full integer quantisation is both small and
  fastest.
* **Explanation fidelity tracks accuracy.** Under LIME, the variant that costs
  least accuracy perturbs the explanation least; every compressed variant agreed
  with the baseline's predicted class on every image tested.

See `../draft.md` sections 3 and 4 for the full argument and the limitations.
