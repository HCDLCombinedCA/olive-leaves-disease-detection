"""Dependency-light helpers shared by every 2D model test script.

Only needs numpy + Pillow, so it imports cleanly inside each per-model
virtualenv without pulling in matplotlib/opencv.

Scripts add the repo root to sys.path and do `from shared.leafviz import ...`.
"""
from __future__ import annotations

import colorsys
import csv
import os
from typing import List, Sequence

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image file as an (H, W, 3) uint8 RGB array."""
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(array: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) uint8 array to `path`, creating parent dirs."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def distinct_colors(n: int, seed: int = 0) -> np.ndarray:
    """Return an (n, 3) uint8 array of visually distinct colors."""
    rng = np.random.default_rng(seed)
    n = max(int(n), 1)
    out = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (i / n + rng.uniform(0, 0.08)) % 1.0
        s = 0.55 + rng.uniform(0.0, 0.35)
        v = 0.75 + rng.uniform(0.0, 0.25)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of boundary pixels (mask minus its 4-neighbour erosion)."""
    m = mask.astype(bool)
    keep = np.ones_like(m)
    keep[1:, :] &= m[:-1, :]
    keep[:-1, :] &= m[1:, :]
    keep[:, 1:] &= m[:, :-1]
    keep[:, :-1] &= m[:, 1:]
    return m & ~keep


def overlay_masks(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
    alpha: float = 0.5,
    draw_contours: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Blend a list of boolean instance masks onto `image` with distinct colors."""
    out = image.astype(np.float32).copy()
    colors = distinct_colors(len(masks), seed=seed)
    hw = image.shape[:2]
    for i, m in enumerate(masks):
        m = np.asarray(m).astype(bool)
        if m.shape != hw:
            continue
        out[m] = (1.0 - alpha) * out[m] + alpha * colors[i].astype(np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    if draw_contours:
        for i, m in enumerate(masks):
            m = np.asarray(m).astype(bool)
            if m.shape != hw:
                continue
            out[_boundary(m)] = colors[i]
    return out


def write_crops(image, masks, crop_dir, stem, all_masks=None):
    """Write per-leaf cutouts + a full overlay into `crop_dir`, return #crops.

    For every mask in `masks`, save a transparent-background PNG (`leaf_000.png`,
    ...) of the leaf cropped to its bounding box, with pixels outside the mask
    made transparent (RGBA). Also save `<stem>_all_masks.png`: the whole image
    with an overlay of `all_masks` if given (e.g. every raw mask a model
    produced), otherwise of `masks` themselves.
    """
    os.makedirs(crop_dir, exist_ok=True)
    overlay_src = masks if all_masks is None else all_masks
    save_image(overlay_masks(image, list(overlay_src)),
               os.path.join(crop_dir, f"{stem}_all_masks.png"))

    count = 0
    for m in masks:
        m = np.asarray(m).astype(bool)
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        rgb = image[y0:y1, x0:x1]
        alpha = m[y0:y1, x0:x1].astype(np.uint8) * 255
        rgba = np.dstack([rgb, alpha]).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(
            os.path.join(crop_dir, f"leaf_{count:03d}.png"))
        count += 1
    return count


def write_masks(masks, mask_dir, stem):
    """Write one full-size binary PNG per instance mask into `mask_dir`.

    Each `mask_NNN.png` is single-channel: 255 (white) for masked pixels, 0
    (black) for background, at the original image resolution (unlike
    `write_crops`, which crops to the bounding box). This full-frame,
    per-instance format is what mask-overlap evaluators (SBD, AP) need to
    match predicted leaves against ground-truth leaf masks. Returns the
    number of masks written.
    """
    os.makedirs(mask_dir, exist_ok=True)
    count = 0
    for m in masks:
        m = np.asarray(m).astype(bool)
        out = (m.astype(np.uint8)) * 255
        Image.fromarray(out, mode="L").save(
            os.path.join(mask_dir, f"mask_{count:03d}.png"))
        count += 1
    return count


def save_counts_csv(path, rows):
    """Write predicted leaf counts to `path` as CSV with header `image,n_leaves`.

    `rows` is an iterable of (image_name, count). Feeds the evaluator in
    eval/evaluate_leaf_count.py, which compares these against ground truth.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_leaves"])
        for name, n in rows:
            w.writerow([name, int(n)])


def masks_from_label_map(label_map: np.ndarray) -> List[np.ndarray]:
    """Split a semantic/instance label map into a list of boolean masks (skip 0)."""
    return [label_map == v for v in np.unique(label_map) if v != 0]


def green_fraction(image: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of masked pixels that look leaf-green (simple RGB heuristic).

    Cheap, dependency-free proxy used to filter SAM's class-agnostic masks down
    to plausible foliage. Not a substitute for a trained classifier.
    """
    m = np.asarray(mask).astype(bool)
    if m.sum() == 0:
        return 0.0
    px = image[m].astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    green = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
    return float(green.mean())
