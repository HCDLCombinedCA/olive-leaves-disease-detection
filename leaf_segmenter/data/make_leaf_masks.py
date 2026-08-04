"""Generate leaf masks for a folder of leaf photos, ready for fine-tuning.

Written for the single-leaf olive photos in `data/olive_leaves/`: one detached
leaf lying on a plain background (white paper, wooden table). Each image gets ONE
leaf instance, so this is a leaf-vs-background annotator, not a multi-leaf
instance segmenter like the CVPPP scripts -- every count it writes is therefore 1.

The masks are produced automatically -- they are good enough to fine-tune on, but
they are not hand-drawn ground truth. Check the contact sheets before training.

How a mask is made
------------------
1. Background model -- the background colour is the median CIELAB value of a band
   around the image edge, so white paper and a wooden table both work with no
   per-image tuning. Every later test is expressed against it.
2. Candidates -- SAM v1 is prompted with a grid of foreground points (plus points
   inside a cheap colour prior, and that prior's box); each prompt returns three
   masks. The colour prior itself -- Otsu on distance-from-background, cleaned up
   -- joins the pool as one more candidate, which is what rescues leaves SAM only
   segments in pieces.
3. Selection -- candidates that are implausible as a leaf are dropped (too small
   or too large, too close in colour to the background, or covering so much of
   the border band that they ARE the background). The rest are ranked by
   `sam_score x solidity x colour separation`, and the winner is taken.
4. Promotion -- on a mottled leaf a disease lesion can outscore the leaf itself
   (compact, and sharply separated from white paper). If a candidate contains the
   winner and is at least twice its area while still scoring nearly as well, the
   winner was a piece of it and that container takes over.
5. Growing -- SAM sometimes returns only part of a leaf, so the winner absorbs
   any overlapping candidate whose ADDED region still looks like the leaf, and
   only while the union stays as compact and as colour-separated as the winner
   was. That distinction matters: a missing leaf tip passes both tests, a
   drop shadow fails them.

Every threshold below is tuned on the 42 olive photos in
`data/olive_leaves/fine_tuning/`; a very different setup (leaves on soil, several
leaves per photo) will need them revisited.

Usage -- run it with the 02_leaf_only_sam venv, which already has
segment_anything, cv2 and the SAM checkpoints:

    leaf_segmenter/models/02_leaf_only_sam/.venv/bin/python \
        leaf_segmenter/data/make_leaf_masks.py

    ... --input-dir some/other/folder --model-type vit_l   # sharper, needs ~4 GB RAM
    ... --no-sam                                           # colour prior only, no torch

`--input-dir` may hold the images directly, or hold one subfolder per class
(`images/Healthy/`, `images/olive_peacock_spot/`, ...); each subfolder becomes a
CVPPP-style subset and the output mirrors that structure. Masks are named after
the image, with a trailing `_rgb` stripped, so `1_rgb.png` -> `1_fg.png` and
`A299.jpg` -> `A299_fg.png`.

Output (CVPPP layout, so the existing training/eval scripts read it as-is):
    <output-dir>/mask/<subset>/<stem>_fg.png               binary 0/255 foreground
    <output-dir>/per_leaf_mask/<subset>/<stem>_label.png   0 = background, 1..N = leaf
    <output-dir>/counts.csv                                image,n_leaves -- every image
    <input-dir>/<subset>/<subset>.csv                      per-class counts, CVPPP GT style
    <output-dir>/qc/<subset>/<stem>_overlay.png            QC: mask edge on the photo
    <output-dir>/qc/contact_sheet_<subset>.png             every overlay in one grid
    <output-dir>/masks.csv                                 per-image stats, review these

Fine-tune Mask R-CNN on the result with:
    cd leaf_segmenter/models/04_mask_rcnn
    python train_mask_rcnn.py --data-dir ../../data/olive_leaves/fine_tuning
(its find_pairs() matches `images/<subset>/<stem>_rgb.png` only, so images named
anything else are skipped until they are renamed or find_pairs is relaxed.)
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from shared.helper import list_images, load_image, save_counts_csv, save_image

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = {
    "vit_b": os.path.join(REPO_ROOT, "models", "02_leaf_only_sam", "checkpoints",
                          "sam_vit_b_01ec64.pth"),
    "vit_l": os.path.join(REPO_ROOT, "models", "02_leaf_only_sam", "checkpoints",
                          "sam_vit_l_0b3195.pth"),
}

BORDER_FRAC = 0.04       # width of the edge band that defines the background colour
GRID = 6                 # GRID x GRID foreground points prompted into SAM
N_FG_POINTS = 5          # extra prompt points taken from inside the colour prior
BOX_PAD = 0.02           # prior box padding, as a fraction of the image size

# what may pass as a leaf
MIN_FRAC, MAX_FRAC = 0.005, 0.60   # plausible share of the frame
MIN_CONTRAST = 25.0                # min mean CIELAB distance from the background
MAX_BORDER_COVER = 0.25            # more of the border band than this = background
CONTRAST_FULL = 60.0               # colour separation is "as good as it gets" here

ATTENTION_HEADS_PER_CHUNK = 4   # 0 disables; see patch_attention_for_low_memory

# promoting the winner to the whole leaf it sits on
CONTAINER_RATIO = 2.0    # a container this much bigger means the winner was a part
CONTAINED_FRAC = 0.9     # ... and it has to hold this much of the winner
PROMOTE_QUALITY = 0.85   # ... while still scoring this share of the winner's quality.
                         # Measured over data/olive_leaves, this cannot be made to
                         # separate: promotions that are right score 0.87-0.92, and
                         # ones that swallow a drop shadow score anywhere from 0.88
                         # to 0.99. Raising it to 0.90 trades a shadow-inflated mask
                         # (IMG_20190729_193220) for a mask that collapses to a
                         # fragment (IMG_20190806_152344), which is worse. 0.85 is
                         # the setting with the fewest bad masks -- not a clean rule,
                         # which is why the QC sheets have to be looked at.

# growing the winner
GROW_PASSES = 3          # additions can chain, so sweep the pool a few times
MIN_ADD_FRAC = 0.01      # ignore additions this small relative to the winner
COLOUR_TOL = 0.5         # addition must sit within this share of the leaf-to-bg gap
MIN_SOLIDITY = 0.75      # a leaf is roughly convex; a leaf + a detached blob is not
KEEP_SOLIDITY = 0.93     # a merge may not cost more than this share of compactness
KEEP_CONTRAST = 0.90     # ... nor this share of the colour separation


# ---- background model ------------------------------------------------------

def background(image):
    """Return (lab image, median background lab, border-band mask).

    The background is whatever fills a band around the edge of the frame, which
    holds for these photos: one leaf, roughly centred, on plain backing.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    band = max(2, int(round(min(h, w) * BORDER_FRAC)))
    ring = np.ones((h, w), dtype=bool)
    ring[band:-band, band:-band] = False
    return lab, np.median(lab[ring], axis=0), ring


# ---- mask clean-up and measurement -----------------------------------------

def _fill_holes(mask):
    """Fill interior holes by re-drawing every external contour solid."""
    out = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color=1, thickness=cv2.FILLED)
    return out.astype(bool)


def _largest_component(mask):
    """Keep only the biggest connected blob; drop specks and detached shadows."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def clean(mask):
    """One leaf, no holes."""
    return _fill_holes(_largest_component(mask))


def measure(mask, lab, bg, ring):
    """Leaf-likeness measurements for one candidate mask (None if it is empty)."""
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours) or 1.0
    return dict(area=area, frac=area / (h * w),
                solidity=min(1.0, area / hull),
                contrast=float(np.linalg.norm(lab[mask] - bg, axis=1).mean()),
                border_cover=float((mask & ring).sum() / ring.sum()))


def plausible(m):
    """Could this mask be the leaf at all?"""
    return (MIN_FRAC <= m["frac"] <= MAX_FRAC
            and m["contrast"] >= MIN_CONTRAST
            and m["border_cover"] <= MAX_BORDER_COVER)


def quality(m, sam_score):
    """Rank plausible candidates: confident, compact, and unlike the background."""
    return sam_score * m["solidity"] * min(m["contrast"] / CONTRAST_FULL, 1.0)


# ---- candidate generation --------------------------------------------------

def colour_prior(image, lab, bg):
    """Rough leaf mask from colour alone: how far each pixel is from the background."""
    h, w = image.shape[:2]
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    _, fg = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # open away speckle, close the gaps disease lesions punch in the blade
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return clean(fg > 0)


def patch_attention_for_low_memory(heads_per_chunk=ATTENTION_HEADS_PER_CHUNK):
    """Make SAM's image encoder run its attention a few heads at a time.

    The encoder's global-attention blocks build one (num_heads, 4096, 4096) score
    matrix -- ~800 MB at float32, doubled while softmax runs -- which pushes peak
    RSS to ~2.8 GB and gets the process OOM-killed on a 3 GB machine. Attention is
    independent per head and softmax normalises each row, so slicing the heads
    into chunks gives bit-identical results with a fraction of the peak.

    Call once after importing segment_anything, before building the model.
    """
    import torch
    from segment_anything.modeling.image_encoder import Attention, add_decomposed_rel_pos

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        out = torch.empty_like(q)
        for i in range(0, q.shape[0], heads_per_chunk):
            sl = slice(i, i + heads_per_chunk)
            attn = (q[sl] * self.scale) @ k[sl].transpose(-2, -1)
            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q[sl], self.rel_pos_h,
                                              self.rel_pos_w, (H, W), (H, W))
            out[sl] = attn.softmax(dim=-1) @ v[sl]
            del attn
        out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4)
        return self.proj(out.reshape(B, H, W, -1))

    Attention.forward = forward


def _prompt_points(mask, k=N_FG_POINTS):
    """Pick up to `k` well-separated points deep inside `mask` (x, y order).

    Deepest-first via the distance transform, then greedily skipping anything too
    close to a point already chosen, so the prompts span the whole leaf instead
    of clustering at its thickest part.
    """
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    order = np.argsort(dt[ys, xs])[::-1]
    min_sep = max(20.0, 0.5 * np.sqrt(mask.sum()))
    picked = []
    for i in order:
        p = np.array([xs[i], ys[i]], dtype=np.float32)
        if all(np.linalg.norm(p - q) >= min_sep for q in picked):
            picked.append(p)
            if len(picked) == k:
                break
    return np.stack(picked)


def sam_candidates(predictor, image, prior):
    """Prompt SAM every which way; return [(mask, sam_score, origin)].

    A grid of single foreground points finds the leaf without being told where it
    is; the prior's points and box add prompts that already know roughly where it
    is. Each prompt yields SAM's three whole-object / part / sub-part masks.
    """
    h, w = image.shape[:2]
    predictor.set_image(image)
    out = []

    points = [(x, y)
              for y in np.linspace(h / (GRID + 1), h - h / (GRID + 1), GRID)
              for x in np.linspace(w / (GRID + 1), w - w / (GRID + 1), GRID)]
    if prior.any():
        points += [tuple(p) for p in _prompt_points(prior)]
    for x, y in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]], dtype=np.float32),
            point_labels=np.array([1]), multimask_output=True)
        out += [(m, float(s), "sam") for m, s in zip(masks, scores)]

    if prior.any():
        ys, xs = np.nonzero(prior)
        box = np.array([max(0, xs.min() - BOX_PAD * w), max(0, ys.min() - BOX_PAD * h),
                        min(w, xs.max() + BOX_PAD * w), min(h, ys.max() + BOX_PAD * h)],
                       dtype=np.float32)
        fg = _prompt_points(prior)
        masks, scores, _ = predictor.predict(
            point_coords=fg, point_labels=np.ones(len(fg), dtype=np.int32),
            box=box, multimask_output=True)
        out += [(m, float(s), "sam-box") for m, s in zip(masks, scores)]
    return out


# ---- selection -------------------------------------------------------------

def promote_to_container(scored, best_quality, base, base_stats):
    """Swap the winner for a candidate that CONTAINS it and is much larger.

    A disease lesion is compact and stands out sharply against white paper, so on
    a mottled leaf it can outscore the leaf it sits on -- and `grow` cannot rescue
    that, because the rest of the blade looks nothing like the lesion. A leaf that
    merely picked up its drop shadow is only ~30% bigger than the leaf, so the
    CONTAINER_RATIO gap keeps this from swallowing shadows; the quality floor
    keeps it from jumping to a big dull patch of background that happens to
    enclose the leaf.

    `scored` is [(quality, mask, stats, origin)]. Returns the entry to use.
    """
    containers = [(m, st, o) for q, m, st, o in scored
                  if st["area"] >= CONTAINER_RATIO * base_stats["area"]
                  and (m & base).sum() >= CONTAINED_FRAC * base.sum()
                  and q >= PROMOTE_QUALITY * best_quality]
    if not containers:
        return None
    return max(containers, key=lambda c: c[1]["area"])


def grow(base, base_stats, pool, lab, bg, ring):
    """Absorb overlapping candidates that extend `base` with more of the same leaf.

    An addition has to look like the leaf (its mean colour close to the leaf's,
    relative to how far the leaf sits from the background) AND leave the union as
    compact and as colour-separated as `base` was. A leaf tip SAM missed passes;
    the leaf's drop shadow fails, because absorbing it dilutes both measures.
    """
    leaf_lab = lab[base].mean(axis=0)
    tol = COLOUR_TOL * float(np.linalg.norm(leaf_lab - bg))
    merges = 0
    for _ in range(GROW_PASSES):
        grew = False
        for cand, _, _ in pool:
            add = cand & ~base
            if add.sum() < MIN_ADD_FRAC * base.sum() or not (cand & base).any():
                continue
            if np.linalg.norm(lab[add].mean(axis=0) - leaf_lab) > tol:
                continue
            grown = clean(base | cand)
            g = measure(grown, lab, bg, ring)
            if (g["solidity"] < MIN_SOLIDITY
                    or g["solidity"] < KEEP_SOLIDITY * base_stats["solidity"]
                    or g["contrast"] < KEEP_CONTRAST * base_stats["contrast"]):
                continue
            base, merges, grew = grown, merges + 1, True
        if not grew:
            break
    return base, merges


def segment_leaf(image, predictor):
    """Return (mask, stats, origin, merges) for the one leaf in `image`.

    `mask` is None when nothing in the image passes as a leaf.
    """
    lab, bg, ring = background(image)
    prior = colour_prior(image, lab, bg)

    pool = []
    if predictor is not None:
        pool += sam_candidates(predictor, image, prior)
    if prior.any():
        # the prior competes on the same terms as SAM's masks; the nominal score
        # keeps it in contention without letting it outrank a confident SAM mask
        pool.append((prior, 0.9, "colour-prior"))

    scored = []
    for mask, sam_score, origin in pool:
        mask = clean(mask)
        stats = measure(mask, lab, bg, ring)
        if stats is not None and plausible(stats):
            scored.append((quality(stats, sam_score), mask, stats, origin))
    if not scored:
        return None, None, "none", 0

    best_quality, mask, stats, origin = max(scored, key=lambda c: c[0])
    promoted = promote_to_container(scored, best_quality, mask, stats)
    if promoted is not None:
        mask, stats, origin = promoted

    mask, merges = grow(mask, stats, [(m, s, o) for _, m, s, o in scored],
                        lab, bg, ring)
    return mask, measure(mask, lab, bg, ring), origin, merges


# ---- outputs ---------------------------------------------------------------

def outline_overlay(image, mask, colour=(255, 40, 40), alpha=0.35):
    """Photo with the mask tinted and its outline drawn, for eyeballing quality."""
    out = image.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, np.float32)
    out = out.clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


def contact_sheet(overlay_paths, path, cols=4, cell=(400, 300)):
    """Tile every overlay into one labelled grid image for a single-glance review."""
    rows = (len(overlay_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(overlay_paths):
        x, y = (i % cols) * cell[0], (i // cols) * (cell[1] + 18)
        sheet.paste(Image.open(p).resize(cell, Image.LANCZOS), (x, y + 18))
        draw.text((x + 4, y + 4), os.path.basename(p).replace("_overlay.png", ""),
                  fill="black")
    sheet.save(path)


def mask_stem(path):
    """Name masks after the image, CVPPP style: `1_rgb.png` -> `1`, `A299.jpg` -> `A299`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[: -len("_rgb")] if stem.endswith("_rgb") else stem


def find_subsets(input_dir, fallback):
    """Return [(subset, [image paths])]: one entry per class subfolder, else one
    entry named `fallback` for images sitting directly in `input_dir`."""
    def images_in(folder):
        try:
            return list_images(folder)          # list_images exits on an empty folder
        except SystemExit:
            return []

    subsets = [(d, images_in(os.path.join(input_dir, d)))
               for d in sorted(os.listdir(input_dir))
               if os.path.isdir(os.path.join(input_dir, d))]
    subsets = [(name, paths) for name, paths in subsets if paths]
    return subsets or [(fallback, list_images(input_dir))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir",
                    default=os.path.join(HERE, "olive_leaves", "fine_tuning", "images"),
                    help="leaf photos to annotate, or one subfolder per class")
    ap.add_argument("--output-dir",
                    help="where mask/ and per_leaf_mask/ go (default: the input's "
                         "parent when it is called images/, else the input itself)")
    ap.add_argument("--subset", default="olive",
                    help="subset name to use when the input has no class subfolders")
    ap.add_argument("--model-type", default="vit_b", choices=sorted(CHECKPOINTS),
                    help="SAM checkpoint to segment with; vit_l is sharper but needs "
                         "~4 GB of RAM on CPU (it gets OOM-killed on a 3 GB box)")
    ap.add_argument("--no-sam", action="store_true",
                    help="skip SAM, keep the colour prior alone (no torch needed)")
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif os.path.basename(input_dir) == "images":
        output_dir = os.path.dirname(input_dir)   # a split root: sit beside images/
    else:
        output_dir = input_dir
    subsets = find_subsets(input_dir, args.subset)

    predictor = None
    if not args.no_sam:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = CHECKPOINTS[args.model_type]
        if not os.path.exists(ckpt):
            sys.exit(f"Missing SAM checkpoint {ckpt} (run 02_leaf_only_sam once to "
                     f"download it, or pass --no-sam)")
        if ATTENTION_HEADS_PER_CHUNK:
            patch_attention_for_low_memory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM {args.model_type} on {device} ...")
        predictor = SamPredictor(sam_model_registry[args.model_type](ckpt).to(device))

    stats_rows, counts = [], []
    for subset, paths in subsets:
        label_dir = os.path.join(output_dir, "per_leaf_mask", subset)
        fg_dir = os.path.join(output_dir, "mask", subset)
        qc_dir = os.path.join(output_dir, "qc", subset)
        for d in (label_dir, fg_dir, qc_dir):
            os.makedirs(d, exist_ok=True)

        print(f"\n--- {subset} ({len(paths)} images) ---")
        overlays, subset_counts = [], []
        for path in paths:
            name = os.path.basename(path)
            stem = mask_stem(path)
            image = load_image(path)
            mask, stats, origin, merges = segment_leaf(image, predictor)
            if mask is None:
                print(f"{name}: nothing plausible as a leaf -- SKIPPED")
                stats_rows.append((subset, name, 0, 0, "", "", "none", 0))
                subset_counts.append((name, 0))
                continue

            # one instance per photo: label 1 = leaf, 0 = background
            label_map = mask.astype(np.uint8)
            Image.fromarray(label_map, mode="L").save(
                os.path.join(label_dir, f"{stem}_label.png"))
            save_image(np.stack([mask.astype(np.uint8) * 255] * 3, -1),
                       os.path.join(fg_dir, f"{stem}_fg.png"))
            ov_path = os.path.join(qc_dir, f"{stem}_overlay.png")
            save_image(outline_overlay(image, mask), ov_path)
            overlays.append(ov_path)

            subset_counts.append((name, int(label_map.max())))
            stats_rows.append((subset, name, stats["area"], round(stats["frac"], 4),
                               round(stats["solidity"], 3), round(stats["contrast"], 1),
                               origin, merges))
            print(f"{name}: {stats['area']} px ({stats['frac']:.1%} of frame), "
                  f"solidity {stats['solidity']:.2f}, from {origin} (+{merges} merged)")

        # per-class counts next to the images, in the CVPPP ground-truth style
        # (`plant006_rgb.png, 15`), which is what eval/--gt-dir reads
        subset_dir = os.path.join(input_dir, subset)
        if os.path.isdir(subset_dir):
            with open(os.path.join(subset_dir, f"{subset}.csv"), "w", newline="") as f:
                csv.writer(f).writerows(subset_counts)
        counts += subset_counts
        if overlays:
            contact_sheet(overlays, os.path.join(output_dir, "qc",
                                                 f"contact_sheet_{subset}.png"))

    save_counts_csv(os.path.join(output_dir, "counts.csv"), counts)
    with open(os.path.join(output_dir, "masks.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "image", "leaf_px", "leaf_frac", "solidity",
                         "contrast", "source", "merges"])
        writer.writerows(stats_rows)
    found = sum(1 for r in stats_rows if r[6] != "none")
    print(f"\nWrote {found}/{len(stats_rows)} masks -> {output_dir}\n"
          f"Check qc/contact_sheet_*.png before fine-tuning on these.")


if __name__ == "__main__":
    main()
