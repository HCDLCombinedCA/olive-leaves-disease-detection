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
python run_hq_sam.py --image ../../data/cvppp/images/A1/plant001_rgb.png --model-type vit_tiny
python run_hq_sam.py --input-dir ../../data/cvppp/images/A1 --model-type vit_b --output outputs/
```

Same output convention and green/size filter flags as folder 01
(`--no-green-filter`, `--min-green`, `--min-area`, `--max-area`,
`--points-per-side`).

## Checkpoints

| `--model-type` | file | notes |
|---|---|---|
| `vit_tiny` | `sam_hq_vit_tiny.pth` | Light HQ-SAM — best fit for 6 GB |
| `vit_b` | `sam_hq_vit_b.pth` | comfortable on 6 GB |
| `vit_l` / `vit_h` | `sam_hq_vit_l/h.pth` | more VRAM |

Auto-downloaded from [`lkeab/hq-sam`](https://huggingface.co/lkeab/hq-sam). If
that fails, grab the file from the [sam-hq releases](https://github.com/SysCV/sam-hq#model-checkpoints)
and pass `--checkpoint /path/to/sam_hq_vit_tiny.pth`.

Sources: [sam-hq repo](https://github.com/SysCV/sam-hq) ·
[paper (NeurIPS 2023)](https://arxiv.org/abs/2306.01567)
