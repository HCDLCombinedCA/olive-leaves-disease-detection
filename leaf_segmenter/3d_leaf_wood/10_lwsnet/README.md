# 10 — LWSNet (3D leaf–wood separation)

**What:** A point-based segmentation network for leaf–wood separation of
individual trees. Reports a strong **average F1 ≈ 97.29%** across eight tree
species of varying size and structure.

**Status:** ⚠️ **Training-oriented / limited public weights.** Published in
*Forests* (MDPI, 2023). Open-source *trained* models in this space are scarce, so
plan to train it (or use PointsToWood, folder 09, which ships weights).

## How to use

1. Read the paper for the architecture and training protocol:
   <https://www.mdpi.com/1999-4907/14/7/1303>
2. Check the paper's "Data Availability" / supplementary section for a code link
   (MDPI papers list it there when released).
3. Train on an annotated leaf–wood point-cloud dataset — e.g. **FOR-instance**
   (UAV LiDAR with stem/branch/foliage labels) or the authors' dataset.

## Recommendation

For an immediate 3D result, prefer **PointsToWood** (folder 09) — it's public and
pretrained. Come back to LWSNet if you need its specific architecture or its
reported per-species accuracy and are willing to train.

Related public 3D options if you end up training:
- **RandLA-Net** benchmark on tropical trees — folder 11.
- General point-cloud frameworks: `Pointcept`, `torch-points3d`.
