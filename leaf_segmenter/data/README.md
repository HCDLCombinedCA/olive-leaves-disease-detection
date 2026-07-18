# Test data

Put your own images / point clouds here — the run scripts point at this folder
by default.

## 2D (RGB photos of trees / branches / leaves)

Drop `.jpg` / `.png` files into `data/samples/`. Then point any 2D script at a
single file or the whole folder:

```bash
python run_*.py --image ../../data/samples/my_branch.jpg --output outputs/
python run_*.py --input-dir ../../data/samples --output outputs/
```

### Don't have photos yet? Make a synthetic one

`make_synthetic_sample.py` draws a fake brown branch with green leaf ellipses so
you can smoke-test the whole pipeline before collecting real data. It only needs
numpy + Pillow.

```bash
python3 make_synthetic_sample.py           # -> data/samples/synthetic_leaf.png
```

This is a sanity check for "does the model run and produce masks", **not** a
benchmark. Zero-shot models (SAM/HQ-SAM) will find blobs; COCO-pretrained models
(Mask R-CNN, YOLO, Mask2Former) have no "leaf" class and will mostly find
nothing until fine-tuned — that is expected.

## 3D (LiDAR / TLS point clouds)

Leaf–wood separation (folder `3d_leaf_wood/`) expects point clouds, typically
`.ply` / `.las` / `.laz`. Put them here and pass the path to the respective
tool. Public sample clouds: **FOR-instance** (UAV LiDAR with stem/branch/foliage
labels) and the datasets linked from each 3D model's README.
