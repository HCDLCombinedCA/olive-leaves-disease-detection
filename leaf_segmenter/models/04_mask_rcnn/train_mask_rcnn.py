"""Fine-tune Mask R-CNN on the CVPPP leaf dataset

Every leaf is treated as one foreground class ("leaf"), so num_classes = 2
(background + leaf), matching NUM_CLASSES in run_mask_rcnn.py. We start from the
COCO-pretrained torchvision model, replace its two prediction heads with 2-class
heads, train on the fine_tuning split, and save the weights as a plain
state_dict that `run_mask_rcnn.py --weights` can load.

Data expected (this is what data/split_dataset.py already produces):
    <data-dir>/images/A?/plantNNN_rgb.png          the photo
    <data-dir>/per_leaf_mask/A?/plantNNN_label.png  0 = background, 1..N = leaf id

Example:
    python train_mask_rcnn.py                       # uses ../../data/cvppp/fine_tuning
    python train_mask_rcnn.py --epochs 20 --output finetuned.pth

Then run it:
    python run_mask_rcnn.py --input-dir ../../data/cvppp/images/A1 --weights finetuned.pth
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import masks_from_label_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # background + leaf (keep in sync with run_mask_rcnn.py)


def find_pairs(data_dir):
    """Return [(image_path, label_path)] by matching each *_rgb.png to its label map."""
    pairs = []
    for img in sorted(glob.glob(os.path.join(data_dir, "images", "*", "*_rgb.png"))):
        # .../images/A1/plant001_rgb.png -> .../per_leaf_mask/A1/plant001_label.png
        label = (img.replace(f"{os.sep}images{os.sep}", f"{os.sep}per_leaf_mask{os.sep}")
                    .replace("_rgb.png", "_label.png"))
        if os.path.exists(label):
            pairs.append((img, label))
    return pairs


class LeafDataset(torch.utils.data.Dataset):
    """Turns each (image, label map) pair into the (image, target) that Mask R-CNN wants."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, label_path = self.pairs[i]
        image = np.array(Image.open(img_path).convert("RGB"))
        label_map = np.array(Image.open(label_path))

        # one boolean mask per leaf, and each leaf's bounding box
        boxes, masks = [], []
        for m in masks_from_label_map(label_map):
            ys, xs = np.where(m)
            if xs.size == 0 or xs.max() == xs.min() or ys.max() == ys.min():
                continue  # skip empty / 1-pixel-thin masks (they make the loss NaN)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(masks), dtype=torch.int64),   # every leaf = class 1
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
        }
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return image, target


def build_model():
    """COCO-pretrained Mask R-CNN with its two heads swapped for NUM_CLASSES outputs."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)
    return model.to(DEVICE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../../data/cvppp/fine_tuning",
                    help="split root holding images/ and per_leaf_mask/")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="keep small on a 6 GB GPU; drop to 1 if you hit OOM")
    ap.add_argument("--output", default="finetuned.pth",
                    help="where to save the trained weights (a plain state_dict)")
    args = ap.parse_args()

    pairs = find_pairs(args.data_dir)
    if not pairs:
        sys.exit(f"No image/label pairs found under {args.data_dir}")
    print(f"Training on {len(pairs)} images, {args.epochs} epochs, on {DEVICE}")

    # images have different sizes, so batches stay as lists (collate = zip)
    loader = torch.utils.data.DataLoader(
        LeafDataset(pairs), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)))

    model = build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        running = 0.0
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # in train mode the model returns a dict of losses; we sum and step
            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"epoch {epoch + 1}/{args.epochs}  mean loss {running / len(loader):.4f}")

    torch.save(model.state_dict(), args.output)  # state_dict, so --weights can load it
    print(f"Saved fine-tuned weights -> {args.output}")


if __name__ == "__main__":
    main()
