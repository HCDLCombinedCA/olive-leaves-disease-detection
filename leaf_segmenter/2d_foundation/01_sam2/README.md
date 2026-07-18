# 01 — SAM 2 (Segment Anything Model 2, Meta)

**What:** Foundation segmentation model. Zero-shot — it segments *everything* in
an image with no training. We run it in "automatic mask" mode and then keep the
green, mid-sized masks as candidate leaves. Best immediate starting point for 2D.

**Status:** ✅ Runs out of the box. Weights auto-download from Hugging Face.

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
# make a throwaway test image if you don't have photos yet
python3 ../../data/make_synthetic_sample.py

python run_sam2.py --image ../../data/samples/synthetic_leaf.png --output outputs/
# whole folder, smaller/faster model:
python run_sam2.py --input-dir ../../data/samples --model-size tiny --output outputs/
# keep every mask SAM finds (no foliage filter):
python run_sam2.py --image ../../data/samples/synthetic_leaf.png --no-green-filter
```

Output: an overlay PNG per image in `outputs/` with each kept mask in a distinct
color, plus a console line `N raw masks -> M kept`.

## Options that matter on a 6 GB GPU

| Flag | Default | Notes |
|---|---|---|
| `--model-size` | `small` | `tiny`/`small` are safe on 6 GB; `large` may OOM. |
| `--points-per-side` | `32` | Drop to `16` for less memory / faster, coarser masks. |
| `--min-green` | `0.5` | Fraction of green pixels required to keep a mask. |
| `--no-green-filter` | off | Keep all class-agnostic masks. |

## Notes

- SAM 2 has no notion of "leaf" — the green/size filter is a cheap heuristic. For
  real accuracy, fine-tune an instance model (folders 04–06) or the forestry
  ones (07–08).
- **SAM 3** now exists (`facebookresearch/sam3`) and adds text/concept prompts
  ("segment the leaves"); swap the model id if you want to try it.

Sources: [SAM 2 repo](https://github.com/facebookresearch/sam2) ·
[paper](https://arxiv.org/abs/2408.00714)
