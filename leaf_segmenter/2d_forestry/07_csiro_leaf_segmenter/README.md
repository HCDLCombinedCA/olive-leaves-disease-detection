# 07 — CSIRO `leaf_segmenter_public` (pretrained leaf Mask R-CNN)

**What:** A Mask R-CNN **pretrained on 10,000 synthetic Arabidopsis images** with
leaf instance labels, released by CSIRO to accompany the CVPPP Leaf Segmentation
Challenge. It's a ready-made *leaf* model (unlike the COCO ones), so it's worth
trying if your subject is rosette/top-down foliage.

**Status:** ⚠️ **Legacy.** Built on Matterport Mask R-CNN + **TensorFlow 1.15 /
Keras 2.2**, which will *not* run on your Python 3.12. It needs an isolated
Python 3.7 environment (conda/pyenv). Weights are a separate download.

**Reference papers:** [Deep Leaf Segmentation Using Synthetic Data (Ward, Moghadam & Hudson, BMVC-W 2018)](https://arxiv.org/abs/1807.10931) — the synthetic-Arabidopsis leaf Mask R-CNN this folder uses; generalized in [UPGen — Scalable learning for bridging the species gap in image-based plant phenotyping (Ward & Moghadam, CVIU 2020)](https://arxiv.org/abs/2003.10757).

## Setup

```bash
cd 2d_forestry/07_csiro_leaf_segmenter
bash setup.sh          # clones the CSIRO repo + Matterport Mask R-CNN
```

Then follow the printed steps to create a Python 3.7 conda env, install TF 1.15,
download the pretrained `.h5` weights from the CSIRO dataset page, and run the
script bundled in `upstream/leaf_segmenter_public`.

- Repo: `https://bitbucket.csiro.au/scm/ag3d/leaf_segmenter_public.git`
- Mirror / related: [DanielCWard/Deep-Leaf-Segmentation-Using-Synthetic-Data](https://github.com/DanielCWard/Deep-Leaf-Segmentation-Using-Synthetic-Data)
- Dataset + weights: [Synthetic Arabidopsis Dataset](https://research.csiro.au/robotics/databases/synthetic-arabidopsis-dataset/)

## Reality check — is it worth it?

The domain is **top-down single Arabidopsis rosettes**, not trees/branches
outdoors. If your images are trees, expect a domain gap. Two better paths for
most people:

1. **Modern, trainable:** fine-tune torchvision Mask R-CNN (folder 04) or
   YOLO11-seg (folder 05) on **Poplar-leaf** (folder 08) — no TF1 pain, better
   fit to outdoor tree foliage.
2. **Zero-shot now:** SAM 2 / HQ-SAM (folders 01, 03).

Use this folder only if you specifically want CSIRO's pretrained leaf weights or
are reproducing the UPGen / synthetic-Arabidopsis line of work
([UPGen](https://github.com/csiro-robotics/UPGen)).
