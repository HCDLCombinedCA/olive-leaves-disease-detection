# 11 — RandLA-Net / pointwise deep learning (tropical TLS leaf–wood)

**What:** Pointwise deep learning for leaf–wood separation of tropical tree point
clouds from TLS. The RandLA-Net benchmark reports **mIoU 86.8% / overall
accuracy 94.8%** on a 148-tree annotated dataset.

**Status:** ⚠️ **Code public, train-your-own.** The QForestLab repository
provides the code from the paper; pretrained weights are limited, so expect to
train on your data (or the authors' dataset).

## Setup

```bash
cd 3d_leaf_wood/11_randlanet_tropical
git clone https://github.com/qforestlab/leaf-wood-segmentation-with-deep-learning.git upstream
# then follow upstream/README for environment + training/inference commands
```

- Code: [qforestlab/leaf-wood-segmentation-with-deep-learning](https://github.com/qforestlab/leaf-wood-segmentation-with-deep-learning)
- Paper: [Pointwise Deep Learning for Leaf-Wood Segmentation of Tropical Tree Point Clouds (ISPRS J., 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0924271625002497)

## Recommendation

Same as folder 10: if you just want a working 3D result now, use **PointsToWood**
(folder 09, pretrained). Choose this one when your subject is **tropical** trees
or you specifically want the RandLA-Net approach and can train.
