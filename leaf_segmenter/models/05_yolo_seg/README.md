# 05 — YOLOv8-seg / YOLO11-seg (Ultralytics)

**What:** Fast, easy, strong instance segmentation. In orchard experiments
YOLOv8-seg beat Mask R-CNN on trunks/branches (mAP@0.5 0.845 vs 0.748), and
YOLO11-seg improves further on occluded objects. This is usually the best
accuracy-for-effort choice once you have labels.

**Status:** ✅ Runs out of the box (COCO weights auto-download). ⚠️ COCO has no
"leaf" class — fine-tune for real results (`train_yolo_seg.py`).

**Reference:** [Ultralytics YOLO (segmentation docs)](https://docs.ultralytics.com/tasks/segment/) — YOLOv8 and YOLO11 are released by Ultralytics without a formal peer-reviewed paper; cite the Ultralytics software/docs. Leaf/orchard application: [YOLOv8-seg orchard study (2023)](https://arxiv.org/abs/2312.07571).

## Install

```bash
cd models/05_yolo_seg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run inference

```bash
# COCO baseline (sanity check)
python run_yolo_seg.py --input-dir ../../data/cvppp/images/A1 --output-dir output/coco/A1
# your fine-tuned model
python run_yolo_seg.py --input-dir ../../data/cvppp/images/A1 --weights runs/segment/train/weights/best.pt --output-dir output/finetuned/A1
```

Three flags only: `--input-dir` (required), `--weights` (COCO `yolo11n-seg.pt` by
default, or your fine-tuned `best.pt`), and an optional `--output-dir`. With
`--output-dir`, each image gets its own folder (colour overlay `<stem>_yoloseg.png`
+ one binary PNG per instance) plus a shared `counts.csv` and `timings.csv`;
`run()` returns the per-instance cutouts. `CONF`, `IMGSZ`, and `DEVICE` are constants at the top of
`run_yolo_seg.py`.

Model sizes: `yolo11n-seg.pt` (nano, fastest) → `s` → `m` → `l` → `x`. On a 6 GB
GPU, `n`/`s` train comfortably; `m` is the practical ceiling.

## Fine-tune on leaves

COCO weights find no leaves; fine-tuning on labelled leaf images is where the
accuracy comes from. The flow is always: **get YOLO-seg labels → point a
`.yaml` at them → train**.

### The files YOLO needs

Ultralytics expects a folder tree with images and label files mirrored under
`images/` and `labels/`, split into `train/` and `val/`:

```
yolo_dataset/
├── images/
│   ├── train/   A1_plant006_rgb.png   A1_plant015_rgb.png   ...
│   └── val/     A2_plant021_rgb.png   ...
├── labels/
│   ├── train/   A1_plant006_rgb.txt   A1_plant015_rgb.txt   ...   ← same stem as the image
│   └── val/     A2_plant021_rgb.txt   ...
└── dataset.yaml
```

Each image has **one `.txt` with the same stem** (`A1_plant006_rgb.png` →
`A1_plant006_rgb.txt`). Every line is one leaf instance as a **normalized
polygon** — `class x1 y1 x2 y2 … xn yn`, all coordinates divided by the image
width/height so they fall in `[0, 1]`, at least 3 points per polygon:

```
# labels/train/A1_plant006_rgb.txt   (class 0 = leaf)
0 0.442 0.351 0.424 0.360 0.408 0.377 0.404 0.392 ...
0 0.611 0.502 0.598 0.517 0.585 0.540 ...
```

(An image with no leaves gets an empty `.txt` — a "background" image.) The
dataset `.yaml` ([`leaf.yaml`](leaf.yaml)) names the split folders and the
classes:

```yaml
path: /abs/path/to/yolo_dataset   # dataset root
train: images/train               # relative to path
val: images/val
names:
  0: leaf
```

### Creating the labels

You need those `.txt` polygons. Depending on what you start from:

- **You already have masks (this repo's CVPPP data).** Convert them with
  [`masks_to_yolo.py`](masks_to_yolo.py): it traces each leaf in the
  `per_leaf_mask/` label maps into a polygon, writes the tree above (prefixing
  `A1_`, `A2_`, … so the subsets' reused `plantNNN` names don't collide), and
  fills in `leaf.yaml`. `fine_tuning/` becomes `train`, `testing/` becomes `val`:

  ```bash
  python masks_to_yolo.py            # ../../data/cvppp -> yolo_dataset/ + leaf.yaml
  ```

- **You have raw images and need to draw labels by hand.** Use a polygon
  annotation tool and **export in "YOLOv8 / YOLO segmentation" format**:
  [Roboflow](https://roboflow.com) (web, easiest export), [CVAT](https://cvat.ai),
  [Label Studio](https://labelstud.io), or [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
  (desktop). This gives the best label quality but is the most work.

- **You want a fast first pass, then hand-correct.** Auto-label with SAM:
  Ultralytics' [`auto_annotate`](https://docs.ultralytics.com/reference/data/annotator/)
  (a detector proposes boxes, SAM fills masks → YOLO-seg `.txt`), or run the
  zero-shot leaf segmenters in this repo (`01_sam2`, `03_hq_sam`,
  `02_leaf_only_sam`) to get masks, then convert. Fix the mistakes in one of the
  GUIs above.

- **You have labels in another format.** COCO/VGG JSON → YOLO-seg with
  Ultralytics' [`JSON2YOLO`](https://github.com/ultralytics/JSON2YOLO).

### Check the labels before training

Always eyeball the converted labels — a bad conversion trains a bad model
silently. [`yolo_dataset/preview_labels.py`](yolo_dataset/preview_labels.py) draws
each image's polygons onto a **copy** (originals untouched) so you can browse them:

```bash
python yolo_dataset/preview_labels.py --out yolo_dataset/preview   # -> yolo_dataset/preview/{train,val}/
feh yolo_dataset/preview/train                                     # or click the PNGs in VS Code
```

### Train

```bash
python train_yolo_seg.py --data leaf.yaml --model yolo11n-seg.pt --epochs 100 --batch 4
```

Weights are written to `runs/segment/train/weights/best.pt`. Keep `--batch`
small (4) on 6 GB VRAM. Ultralytics also drops label-overlay previews
(`runs/segment/train/train_batch0.jpg`) and validation metrics as it goes.

Sources: [Ultralytics docs](https://docs.ultralytics.com/tasks/segment/) ·
[YOLOv8 orchard study](https://arxiv.org/abs/2312.07571)
