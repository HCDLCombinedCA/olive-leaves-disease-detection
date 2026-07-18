# 09 — PointsToWood (3D leaf–wood separation) ✅ ready

**What:** Deep-learning framework that classifies every point in a TLS/LiDAR
point cloud as **leaf (0)** or **wood (1)**, from tree base to branch tips. Built
on PointNet/PointNeXt ideas, trained on diverse mature European forests, and it
**ships pretrained weights** — the most ready-to-run option for 3D.

This is *leaf–wood separation* (foliage vs. woody structure), not individual-leaf
instances. If you have LiDAR/TLS and want to strip foliage from branches, start
here.

**Status:** ✅ Public code + pretrained weights.

## Setup

```bash
cd 3d_leaf_wood/09_pointstowood
bash setup.sh          # clones harryjfowen/PointsToWood into upstream/
```

Then, in a dedicated venv (it needs torch + torch-geometric):

```bash
cd upstream/PointsToWood
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python predict.py --point-cloud ../../../../data/your_plot.ply
```

Output: the cloud with per-point columns for prediction (0 = leaf, 1 = wood) and
probability of wood (0.0–1.0).

## Notes for your 6 GB GPU

Point-cloud nets process in tiles/batches; large plots may need smaller batch or
CPU fallback. See the repo README for the batch/voxel flags. `torch-geometric`
wheels must match your torch+CUDA — use the
[PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if pip can't resolve them.

Sources: [repo](https://github.com/harryjfowen/PointsToWood) ·
[paper](https://arxiv.org/abs/2503.04420)
