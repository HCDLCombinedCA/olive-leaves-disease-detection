# 04 — Mask R-CNN

**What:** The classic instance-segmentation workhorse and the long-time standard
for the CVPPP leaf challenge. This folder uses the **torchvision** implementation
with COCO weights because it installs cleanly (no Detectron2 compilation).

**Status:** ✅ Runs out of the box, but ⚠️ **COCO has no "leaf" class** — out of
the box it detects generic objects and usually finds nothing on a bare branch.
Use it as a runnable baseline and as the backbone you **fine-tune** on leaves.

**Reference paper:** [Mask R-CNN (He, Gkioxari, Dollár & Girshick, ICCV 2017)](https://arxiv.org/abs/1703.06870)

## Install & run (COCO baseline / sanity check)

```bash
cd 2d_instance_seg/04_mask_rcnn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --output-dir output/coco/A1
```

Three flags only: `--input-dir` (required), `--weights` (a fine-tuned `.pth`;
omit for COCO weights), and an optional `--output-dir`. With `--output-dir`, each
image gets its own folder (colour overlay `<stem>_maskrcnn.png` + one binary PNG
per instance) plus a shared `counts.csv`; `run()` returns the per-instance
cutouts. Detection/mask thresholds and `NUM_CLASSES` are constants at the top of
`run_mask_rcnn.py`.

## Fine-tune on leaves (where the accuracy comes from)

1. Get an annotated leaf dataset (CVPPP, MSU-PID, or **Poplar-leaf** for trees —
   see folder `2d_forestry/08_leafinst_poplar`).
2. Train with the official torchvision detection reference
   ([`references/detection`](https://github.com/pytorch/vision/tree/main/references/detection))
   or any Mask R-CNN trainer, using `num_classes = 1 background + N leaf classes`.
3. Run your model here:

```bash
python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --weights finetuned.pth
```

(If your fine-tuned model has a different class count, set `NUM_CLASSES` at the
top of `run_mask_rcnn.py`; it defaults to 2 = background + leaf.)

## Alternatives

- **Detectron2** (`facebookresearch/detectron2`): the other common Mask R-CNN.
  More features, but it compiles CUDA extensions and can be fiddly on Python 3.12
  / newer torch. Prefer it only if you already have a working Detectron2 setup.
- **CSIRO `leaf_segmenter_public`**: a Mask R-CNN pretrained on synthetic
  Arabidopsis — see folder `2d_forestry/07_csiro_leaf_segmenter`.

Sources: [Mask R-CNN paper](https://arxiv.org/abs/1703.06870) ·
[torchvision detection ref](https://github.com/pytorch/vision/tree/main/references/detection)
