"""Generate a synthetic 'branch with leaves' image for smoke-testing.

This lets you run every 2D pipeline end-to-end before you have real photos.
It is NOT a benchmark image -- replace it with your own tree/branch photos in
data/samples/ for meaningful results.

Usage:
    python3 make_synthetic_sample.py            # writes data/samples/synthetic_leaf.png
    python3 make_synthetic_sample.py --out foo.png --leaves 40
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def make(width: int, height: int, n_leaves: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    # sky-ish / soil-ish background gradient
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:, :, 0] = np.linspace(120, 90, height)[:, None]
    bg[:, :, 1] = np.linspace(110, 80, height)[:, None]
    bg[:, :, 2] = np.linspace(100, 70, height)[:, None]
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    # a brown branch (thick diagonal line with a few offshoots)
    branch_color = (96, 66, 42)
    x0, y0 = int(width * 0.1), int(height * 0.9)
    x1, y1 = int(width * 0.7), int(height * 0.15)
    draw.line([(x0, y0), (x1, y1)], fill=branch_color, width=max(6, width // 60))
    for t in np.linspace(0.2, 0.9, 5):
        bx = int(x0 + (x1 - x0) * t)
        by = int(y0 + (y1 - y0) * t)
        ex = bx + int(rng.uniform(-1, 1) * width * 0.25)
        ey = by - int(rng.uniform(0.1, 0.3) * height)
        draw.line([(bx, by), (ex, ey)], fill=branch_color, width=max(3, width // 110))

    # green leaf ellipses with slight color variation
    for _ in range(n_leaves):
        cx = int(rng.uniform(0.1, 0.95) * width)
        cy = int(rng.uniform(0.05, 0.9) * height)
        rx = int(rng.uniform(0.02, 0.06) * width)
        ry = int(rx * rng.uniform(1.4, 2.2))
        angle = rng.uniform(0, 360)
        g = int(rng.uniform(110, 190))
        color = (int(g * rng.uniform(0.2, 0.5)), g, int(g * rng.uniform(0.15, 0.4)))
        leaf = Image.new("RGBA", (rx * 2 + 4, ry * 2 + 4), (0, 0, 0, 0))
        ld = ImageDraw.Draw(leaf)
        ld.ellipse([2, 2, rx * 2, ry * 2], fill=color + (255,))
        leaf = leaf.rotate(angle, expand=True)
        img.paste(leaf, (cx - leaf.width // 2, cy - leaf.height // 2), leaf)

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "samples", "synthetic_leaf.png"))
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--leaves", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make(args.width, args.height, args.leaves, args.seed).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
