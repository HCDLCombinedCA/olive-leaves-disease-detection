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
cd models/04_mask_rcnn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --output-dir output/coco/A1
```

Three flags only: `--input-dir` (required), `--weights` (a fine-tuned `.pth`;
omit for COCO weights), and an optional `--output-dir`. With `--output-dir`, each
image gets its own folder (colour overlay `<stem>_maskrcnn.png` + one binary PNG
per instance) plus a shared `counts.csv` and `timings.csv`; `run()` returns the
per-instance cutouts. Detection/mask thresholds and `NUM_CLASSES` are constants at the top of
`run_mask_rcnn.py`.

## Fine-tune on leaves (where the accuracy comes from)

`train_mask_rcnn.py` is a small, self-contained trainer (no Detectron2, no
torchvision `references/detection` copies — just the deps already in
`requirements.txt`). It reads the CVPPP `fine_tuning/` split directly: each
`images/A?/plantNNN_rgb.png` paired with its per-leaf label map
`per_leaf_mask/A?/plantNNN_label.png`, treating **every leaf as one class**
(`num_classes = 2` = background + leaf).

```bash
# from data/split_dataset.py you already have data/cvppp/fine_tuning/
python train_mask_rcnn.py --epochs 20            # -> finetuned.pth
python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --weights finetuned.pth
```

Flags: `--data-dir` (default `../../data/cvppp/fine_tuning`), `--epochs`,
`--batch-size` (default 2; drop to 1 if you OOM on 6 GB), `--output`. It starts
from the COCO-pretrained model, swaps the box + mask heads to 2 classes, trains,
and saves a plain `state_dict` — exactly what `run_mask_rcnn.py --weights` loads.
The script prints the mean loss per epoch; there's no held-out validation, so use
the `eval/` scripts against a `testing/` split to measure accuracy.

Prefer a different dataset (e.g. MSU-PID, or your own annotated leaves)? Arrange
it in the same `images/` + `per_leaf_mask/` layout, or train with the official
torchvision detection reference
([`references/detection`](https://github.com/pytorch/vision/tree/main/references/detection)).
If your model has a different class count, set `NUM_CLASSES` at the top of **both**
`train_mask_rcnn.py` and `run_mask_rcnn.py` (both default to 2).

## Alternatives

- **Detectron2** (`facebookresearch/detectron2`): the other common Mask R-CNN.
  More features, but it compiles CUDA extensions and can be fiddly on Python 3.12
  / newer torch. Prefer it only if you already have a working Detectron2 setup.

Sources: [Mask R-CNN paper](https://arxiv.org/abs/1703.06870) ·
[torchvision detection ref](https://github.com/pytorch/vision/tree/main/references/detection)
