# 01 — SAM 2 (Segment Anything Model 2, Meta)

**What:** Foundation segmentation model. Zero-shot — it segments *everything* in
an image with no training. We run it in "automatic mask" mode and then keep the
green, mid-sized masks as candidate leaves. Best immediate starting point for 2D.

**Status:** ✅ Runs out of the box. Weights auto-download from Hugging Face.

**Reference paper:** [SAM 2: Segment Anything in Images and Videos (Ravi et al., Meta, 2024)](https://arxiv.org/abs/2408.00714)

## Install

```bash
cd 2d_foundation/01_sam2
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If `pip install sam2` fails, install from source instead:
> `pip install "git+https://github.com/facebookresearch/sam2.git"`

## Run

```bash
python run_sam2.py --input-dir ../../data/cvppp/images/A1
# smaller/faster model, and write outputs to disk:
python run_sam2.py --input-dir ../../data/cvppp/images/A1 --model-size tiny --output-dir output/tiny/A1
```

Only three flags: `--input-dir` (required), `--model-size`
(`tiny`/`small`/`base-plus`/`large`, default `small`), and an optional
`--output-dir`. With `--output-dir`, each input image gets its **own folder**
holding the colour overlay (`<stem>_sam2.png`) and one binary PNG per leaf
(`mask_NNN.png`), plus a shared `counts.csv` (`image,n_leaves`) and `timings.csv`
(`image,inference_s,postprocess_s`) at the output root. The `run()` function always returns, per input image, the list of
transparent-background leaf cutouts. Console shows `N raw masks -> M kept` per
image.

Prefer to explore step by step? `run_sam2.ipynb` is a linear notebook version of
the same pipeline (segment → filter → visualise inline), handy for tuning the
green/size thresholds on one image.

## How it works

### The pretrained model

SAM 2 is a *foundation* segmentation model from Meta, pretrained at scale to
segment **any** object from a visual prompt (a click point, a box, or a coarse
mask). It is **class-agnostic** — it has no concept of "leaf", or of any
category; it only finds coherent object regions. The weights are already trained
and **auto-download from Hugging Face** on first run
(`facebook/sam2.1-hiera-{tiny,small,base-plus,large}`, chosen by `--model-size`).
Nothing is trained locally — this is pure zero-shot inference.

### The code pipeline

`run_sam2.py` turns that general model into per-leaf masks in three stages:

1. **Segment everything.** `build_mask_generator()` wraps the pretrained weights
   in `SAM2AutomaticMaskGenerator`. For each image, `generate()` lays down a
   regular grid of `POINTS_PER_SIDE` × `POINTS_PER_SIDE` prompt points
   (32×32 = 1024 by default), prompts SAM at each one as if you had clicked
   there, keeps the confident and stable masks, and removes near-duplicates —
   returning one class-agnostic mask per region it found (leaves, but also pot,
   soil, and background).

2. **Keep the leaf-like masks.** `filter_leaf_masks()` drops masks that are too
   small (`MIN_AREA`, speckle/noise) or too large (`MAX_AREA`, usually
   background or the whole pot), then keeps only masks whose pixels are mostly
   green, via `green_fraction()` in `shared/helper.py` — a cheap RGB rule
   (`g > r*1.05 and g > b*1.05 and g > 40`) thresholded at `MIN_GREEN`. Every
   surviving mask is treated as one leaf instance.

3. **Write outputs.** With `--output-dir`, a colour overlay PNG plus one binary
   PNG per leaf in a per-image folder, and a shared `counts.csv` and
   `timings.csv` (per-image inference / post-processing seconds). `run()` also
   returns the per-leaf transparent-background cutouts in memory
   (`crop_leaves()` in `shared/helper.py`).

Unlike model 02 (Leaf Only SAM), this script applies **only** the area + green
filters — there is no containment de-duplication step — so overlapping rosette
leaves stay as separate masks instead of collapsing into a single whole-plant
blob.

## Options that matter on a 6 GB GPU

`--model-size` is the only tuning knob left on the CLI (`tiny`/`small` are safe on
6 GB; `large` may OOM). The rest are constants at the top of `run_sam2.py`, edit
there if needed:

| Constant | Default | Notes |
|---|---|---|
| `POINTS_PER_SIDE` | `32` | Drop to `16` for less memory / faster, coarser masks. |
| `MIN_GREEN` | `0.5` | Fraction of green pixels required to keep a mask (set `0` to keep all). |
| `MIN_AREA` / `MAX_AREA` | `0.0005` / `0.25` | Drop masks below/above this fraction of the image. |

## Notes

- SAM 2 has no notion of "leaf" — the green/size filter is a cheap heuristic. For
  real accuracy, fine-tune an instance model (folders 04–06) or the forestry
  ones (07–08).
- **SAM 3** now exists (`facebookresearch/sam3`) and adds text/concept prompts
  ("segment the leaves"); swap the model id if you want to try it.

Sources: [SAM 2 repo](https://github.com/facebookresearch/sam2) ·
[paper](https://arxiv.org/abs/2408.00714)
