# 03 — HQ-SAM (Segment Anything in High Quality)

**What:** SAM with a high-quality mask decoder that produces sharper, more
accurate boundaries — helpful for thin, serrated leaf edges where plain SAM
looks blobby. Still zero-shot. `vit_tiny` is **Light HQ-SAM** (TinyViT backbone,
~41 FPS, tiny memory footprint) and is the recommended default on a 6 GB GPU.

**Status:** ✅ Runs out of the box. Checkpoints auto-download from the HF hub.

**Reference paper:** [Segment Anything in High Quality (Ke et al., NeurIPS 2023)](https://arxiv.org/abs/2306.01567)

## Install

```bash
cd 2d_foundation/03_hq_sam
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python run_hq_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_tiny
python run_hq_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b --output-dir output/vit_b/A1
```

Three flags only: `--input-dir` (required), `--model-type` (default `vit_tiny`),
and an optional `--output-dir`. Same output convention as folder 01 — a per-image
folder with the colour overlay (`<stem>_hqsam.png`) + one binary PNG per leaf, a
shared `counts.csv`, and per-leaf cutouts returned by `run()`. The green/size
filter thresholds (`MIN_GREEN`, `MIN_AREA`, `MAX_AREA`, `POINTS_PER_SIDE`) are
constants at the top of `run_hq_sam.py`.

## Checkpoints

| `--model-type` | file | notes |
|---|---|---|
| `vit_tiny` | `sam_hq_vit_tiny.pth` | Light HQ-SAM — best fit for 6 GB |
| `vit_b` | `sam_hq_vit_b.pth` | comfortable on 6 GB |
| `vit_l` / `vit_h` | `sam_hq_vit_l/h.pth` | more VRAM |

Auto-downloaded from [`lkeab/hq-sam`](https://huggingface.co/lkeab/hq-sam). If
that fails, grab the file from the [sam-hq releases](https://github.com/SysCV/sam-hq#model-checkpoints)
and place it in your Hugging Face cache (or edit `HF_REPO` / `CKPT_FILES` in the
script to point at a local copy).

Sources: [sam-hq repo](https://github.com/SysCV/sam-hq) ·
[paper (NeurIPS 2023)](https://arxiv.org/abs/2306.01567)
