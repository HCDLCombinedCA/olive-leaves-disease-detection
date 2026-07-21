# Leaf Instance Segmentation — Evaluation Report (CVPPP A1)

**Date:** 2026-07-21
**Benchmark:** CVPPP A1 subset — 128 top-down RGB images of *Arabidopsis* rosettes
**Ground truth:** 2,088 annotated leaves total (~16.3 leaves/image)
**Evaluators:** `eval/evaluate_leaf_count.py` (LCC), `eval/evaluate_leaf_segmentation.py` (LSC — FBD & SBD)
**Models evaluated:** 5 of 6 — `03_hq_sam` skipped (empty output; OOM during generation, still being fixed)

---

## 1. Metrics at a glance

### Leaf counting (CVPPP LCC)
Lower `|DiC|` / `MSE` is better; higher `%agree` is better. `DiC` is signed bias (pred − gt).

| Rank | Model | DiC (bias) | \|DiC\| | MSE | RMSE | Agreement |
|---|---|---|---|---|---|---|
| 1 | **01 SAM2** | −4.27 (±2.81) | **4.56** (±2.33) | **26.16** | **5.12** | **5.5%** (7/128) |
| 2 | **02 Leaf-Only SAM** | −5.86 (±4.99) | 5.98 (±4.84) | 59.23 | 7.70 | 3.9% (5/128) |
| 3 | 04 Mask R-CNN | −13.86 (±1.90) | 13.86 | 195.67 | 13.99 | 0.0% |
| 4 | 05 YOLO-seg | −14.56 (±1.94) | 14.56 | 215.83 | 14.69 | 0.0% |
| 5 | 06 Mask2Former | −15.59 (±1.98) | 15.59 | 246.82 | 15.71 | 0.0% |

### Leaf segmentation (CVPPP LSC)
Higher is better for FBD and SBD.

| Rank | Model | FBD | SBD | Leaves predicted |
|---|---|---|---|---|
| 1 | **01 SAM2** | 0.848 | **0.605** | 1,541 |
| 2 | **02 Leaf-Only SAM** | **0.946** | 0.579 | 1,338 |
| 3 | 04 Mask R-CNN | 0.422 | 0.058 | 314 |
| 4 | 05 YOLO-seg | 0.290 | 0.033 | 224 |
| 5 | 06 Mask2Former | 0.223 | 0.016 | 93 |

**Metric key:** *FBD* = Foreground-Background Dice (did it find the plant?); *SBD* = Symmetric Best Dice (did it split individual leaves correctly? — the headline CVPPP number).

---

## 2. The single most important finding

**There are two distinct populations of models here, and the gap between them is not a tuning gap — it is a "does the model even know what a leaf is" gap.**

- **Zero-shot foundation models (01 SAM2, 02 Leaf-Only SAM)** are *class-agnostic*: they segment every coherent region in the image. On a rosette, individual leaves *are* coherent regions, so these models produce genuinely useful leaf masks with **no training** (SBD ~0.58–0.61).
- **COCO-pretrained instance models (04 Mask R-CNN, 05 YOLO-seg, 06 Mask2Former)** are *class-conditional* and were trained on the 80 COCO categories, **none of which is "leaf" or "plant."** Out of the box they have nothing to fire on. Their scores (SBD ≤ 0.06, essentially zero for YOLO/Mask2Former) confirm the caveat recorded in `CLAUDE.md`: these are **fine-tuning backbones, not runnable leaf baselines.**

Reading their near-zero results as "these models are bad" would be the wrong conclusion. They are **untrained for this task**. The correct comparison is SAM2 vs Leaf-Only SAM; the COCO trio establishes only a floor and a fine-tuning starting point.

---

## 3. Critical evaluation per model

### 01 — SAM2 · **Best overall**
The strongest all-round performer and the model to beat. It leads on every count metric (`|DiC|` 4.56, MSE 26.2, RMSE 5.12) and on the headline instance-segmentation metric: **SBD 0.605**, the best leaf-splitting quality of any model here.

- **Strength:** best balance of finding leaves *and* separating them. It predicts 1,541 leaves vs 2,088 GT — the closest count of any model — and its high SBD shows those masks align well with individual GT leaves, i.e. it misses the fewest real leaves.
- **Weakness:** still under-segments (DiC −4.27, ~547 fewer leaves than GT). The shortfall is almost certainly the small, young leaves clustered at the rosette centre and heavily-overlapping leaf pairs it merges into one mask. Its FBD (0.848) is notably *lower* than Leaf-Only SAM's — its whole-plant coverage is slightly leakier (it grabs some background / merges leaf-with-shadow), even though its per-leaf splitting is better.
- **Verdict:** production-viable zero-shot for rosette imagery; the under-count of small central leaves is the main thing fine-tuning or prompt/point-density tuning would target.

### 02 — Leaf-Only SAM · **Best foreground, most conservative**
A close second, with a distinct and revealing profile: the **best FBD (0.946)** but the most conservative leaf count and a slightly lower SBD (0.579) than SAM2.

- **Strength:** exceptional at delineating the plant-vs-background boundary (FBD 0.946) and very clean masks — it rarely paints background as leaf, giving it the tightest whole-plant foreground of any model.
- **Weakness:** it is **too conservative.** It predicts only 1,338 leaves (750 fewer than GT), the worst under-count among the SAM models (DiC −5.86) and the **highest count variance (±4.99)** — its error is inconsistent image-to-image. Its leaf-filtering step (the "leaf-only" heuristic that gives it clean foreground) is also what suppresses genuine small/ambiguous leaves, so it misses more real leaves than SAM2.
- **Verdict:** excellent when a clean plant mask matters more than counting every leaf (e.g. total leaf-area estimation); inferior to SAM2 when the goal is an accurate leaf *count* or catching every instance.

### 04 — Mask R-CNN (COCO) · **Detects objects, none of them leaves**
Best of the COCO trio, but that is a low bar. FBD 0.422 shows it fires on *something* covering ~42% of the plant blob, and it emits 314 detections — but its SBD of 0.058 shows almost none of them align with an individual leaf.

- **Interpretation:** it is confidently detecting COCO-shaped blobs (likely the pot, the whole rosette-as-one-object, soil texture) that partially cover the plant but do not correspond to individual leaves — mostly wrong regions.
- **Verdict:** unusable as-is; **most promising fine-tuning candidate** of the three because its region-proposal machinery already localises *some* plant structure. Needs training on leaf annotations (see `05_yolo_seg/train_yolo_seg.py` pattern).

### 05 — YOLO-seg (COCO) · **Nothing aligns with a leaf**
FBD 0.290, and its SBD of 0.033 means effectively none of its 224 predictions align with a real leaf. It detects fewer, lower-quality regions than Mask R-CNN.

- **Interpretation:** its 224 detections are COCO false-alarms with no meaningful leaf overlap. The severe count under-shoot (DiC −14.56) reflects how little it finds.
- **Verdict:** unusable zero-shot; a fine-tuning backbone only. The repo already ships the training scaffolding (`train_yolo_seg.py` + `leaf.yaml`) precisely because this model is expected to require it.

### 06 — Mask2Former (COCO) · **Detects the least of all**
The weakest across the board: FBD 0.223, SBD 0.016, and the largest count bias (DiC −15.59). It emits only 93 detections — it is the most conservative COCO model and finds almost nothing on plant imagery.

- **Interpretation:** the Swin-base COCO-instance checkpoint produces very few high-confidence masks on out-of-distribution rosette images, and none align with leaves.
- **Verdict:** unusable zero-shot; heaviest model here yet lowest yield without training. Strong architecture, wrong weights for this task.

---

## 4. Cross-cutting key findings

1. **Every model under-counts (all DiC < 0).** No model over-segments. For the SAM models this is a modest, correctable bias (−4 to −6 leaves/image) concentrated on small overlapping leaves; for the COCO models it reflects near-total failure to detect (−14 to −16 leaves/image).

2. **Counting quality and segmentation quality agree.** The count ranking (SAM2 > Leaf-Only SAM > COCO trio) exactly matches the SBD ranking, and the two evaluators are internally consistent (e.g. SAM2's 1,541 predicted leaves ≈ 2,088 GT + DiC(−4.27) × 128 images). This is a good cross-check that the pipeline and ground-truth matching are wired correctly.

3. **FBD and SBD measure different things — and the two SAM models trade off between them.** Leaf-Only SAM wins "find the plant" (FBD 0.946) while SAM2 wins "separate the leaves" (SBD 0.605). A model can find the plant almost perfectly yet still be mediocre at instance separation. For leaf *counting* and per-leaf phenotyping, **SBD is the metric that matters**, so SAM2 is the better default.

4. **Missed leaves, not spurious masks, are the bottleneck for the SAM models.** Both under-count (DiC −4.27 and −5.86) and predict fewer leaves than exist (1,541 and 1,338 vs 2,088 GT), so their error is dominated by *missed* leaves rather than extra false masks. Improvement effort should target the small/occluded central leaves they miss (denser point sampling, multi-scale prompting, or fine-tuning).

---

## 5. Caveats & limitations

- **One subset only.** Results are A1 (*Arabidopsis* rosettes) exclusively. A3 (tobacco, larger leaves) and A4 would likely shift the numbers, especially for the SAM models. A1/A2/A3 reuse `plantNNN` filenames, so subsets must be evaluated one at a time.
- **Domain mismatch vs the project's real target.** This harness is intended for olive/forestry imagery (photos of branches and trees), but CVPPP A1 is controlled top-down potted rosettes. A1 is a clean, well-annotated **sanity benchmark** — strong A1 numbers do not guarantee equivalent performance on cluttered outdoor olive foliage, where leaf overlap, background, and lighting are far harsher.
- **03 HQ-SAM is missing.** Its output directory is empty following an OOM during generation. The evaluation should be re-run once it produces `_masks/` folders and a `counts.csv`; based on its lineage it is expected to land near SAM2, potentially with cleaner mask boundaries.

---

## 6. Recommendations

1. **Default to SAM2 (01)** for zero-shot leaf segmentation and counting on this style of imagery. Use **Leaf-Only SAM (02)** when a clean whole-plant foreground matters more than catching every leaf.
2. **Do not deploy 04/05/06 as-is.** Treat them as fine-tuning backbones. If a supervised model is wanted, **Mask R-CNN (04)** is the most promising starting point (it already localises some plant structure); the YOLO training scaffold in the repo is the fastest path to a first trained model.
3. **Finish and evaluate 03 HQ-SAM** (fix the OOM: smaller variant / `--points-per-side 16` / lower image size / `--device cpu`, per `CLAUDE.md`), then re-run both evaluators for a complete comparison.
4. **Extend beyond A1** to A3/A4 before drawing firm conclusions, and ultimately validate on real olive imagery, since that is the project's actual target domain.
