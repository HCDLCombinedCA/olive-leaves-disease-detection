# 08 — LeafInst + Poplar-leaf dataset (forestry, most on-point)

**What:** The work closest to "segment leaves on tree branches outdoors."

- **Poplar-leaf**: the *first* pixel-level instance-segmentation dataset for
  forestry leaves in open-field scenes — **1,202 young poplar branches, 19,876
  leaf instances**, collected by UAV.
- **LeafInst**: a segmentation network tailored to irregular, multi-scale leaves
  (AFPN + DASP + DARH modules). Reports **68.4 mAP on Poplar-leaf**, beating
  YOLOv11 by 7.1 and MaskDINO by 6.5.

**Reference paper:** [LeafInst + Poplar-leaf dataset (March 2026)](https://arxiv.org/abs/2603.03616)

**Status:** ⏳ **No public code or weights released yet** (as of this writing —
the arXiv paper links neither a repo nor a dataset mirror). So there is nothing
to `pip install` and run today. Two ways forward:

## Option A — watch for the release, then run it

Check these for a code/dataset drop, then follow their instructions:

- The paper's arXiv page (authors usually add a "Code:" link on revision):
  <https://arxiv.org/abs/2603.03616>
- Author pages / Papers-with-Code:
  <https://paperswithcode.com/search?q=LeafInst>

When the Poplar-leaf dataset appears, it will be the single best data source for
every trainable model in this repo.

## Option B — get results now on the same problem (recommended)

LeafInst benchmarks *against* YOLOv11-seg, which you already have in folder 05.
So the practical path today:

1. Obtain a tree-leaf instance dataset — Poplar-leaf when released, or annotate a
   small set of your own branch photos (Roboflow / CVAT export to YOLO-seg
   polygons).
2. Fine-tune YOLO11-seg on it:

   ```bash
   cd ../../2d_instance_seg/05_yolo_seg
   # edit leaf.yaml -> your dataset, then:
   python train_yolo_seg.py --data leaf.yaml --model yolo11s-seg.pt --epochs 100 --batch 4
   ```

3. You'll be within a few mAP of LeafInst's reported numbers, on runnable code.

For canopy-scale aerial work (whole tree crowns rather than individual leaves),
see **YOLOv8E** for tree-crown instance segmentation as a related option.
