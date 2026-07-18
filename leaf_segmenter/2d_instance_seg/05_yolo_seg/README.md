# 05 — YOLOv8-seg / YOLO11-seg (Ultralytics)

**What:** Fast, easy, strong instance segmentation. In orchard experiments
YOLOv8-seg beat Mask R-CNN on trunks/branches (mAP@0.5 0.845 vs 0.748), and
YOLO11-seg improves further on occluded objects. This is usually the best
accuracy-for-effort choice once you have labels.

**Status:** ✅ Runs out of the box (COCO weights auto-download). ⚠️ COCO has no
"leaf" class — fine-tune for real results (`train_yolo_seg.py`).

## Install

```bash
cd 2d_instance_seg/05_yolo_seg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run inference

```bash
# COCO baseline (sanity check)
python run_yolo_seg.py --image ../../data/samples/synthetic_leaf.png --output outputs/
# your fine-tuned model
python run_yolo_seg.py --input-dir ../../data/samples --weights runs/segment/train/weights/best.pt
```

Model sizes: `yolo11n-seg.pt` (nano, fastest) → `s` → `m` → `l` → `x`. On a 6 GB
GPU, `n`/`s` train comfortably; `m` is the practical ceiling.

## Fine-tune on leaves

1. Put your dataset in Ultralytics YOLO-seg polygon format and edit
   [`leaf.yaml`](leaf.yaml) with the paths + class names.
2. Train:

```bash
python train_yolo_seg.py --data leaf.yaml --model yolo11n-seg.pt --epochs 100 --batch 4
```

Weights are written to `runs/segment/train/weights/best.pt`. Keep `--batch`
small (4) on 6 GB VRAM.

**Converting other formats:** if your labels are COCO/VGG/masks, convert to
YOLO-seg with Ultralytics' `JSON2YOLO` tools or Roboflow export. The
**Poplar-leaf** forestry dataset (folder 08) is a good source of tree-leaf labels.

Sources: [Ultralytics docs](https://docs.ultralytics.com/tasks/segment/) ·
[YOLOv8 orchard study](https://arxiv.org/abs/2312.07571)
