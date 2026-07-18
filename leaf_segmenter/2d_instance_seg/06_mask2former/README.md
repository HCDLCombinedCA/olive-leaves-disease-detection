# 06 — Mask2Former

**What:** Transformer-based universal segmentation, state-of-the-art on the CVPPP
leaf-segmentation benchmark. Run here via Hugging Face transformers with
COCO-instance weights (auto-download).

**Status:** ✅ Runs out of the box. ⚠️ COCO weights have no "leaf" class —
fine-tune for real leaf results, or pass a fine-tuned checkpoint with `--model`.

## Install & run

```bash
cd 2d_instance_seg/06_mask2former
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_mask2former.py --image ../../data/samples/synthetic_leaf.png --output outputs/
python run_mask2former.py --input-dir ../../data/samples
```

## Model choice on 6 GB VRAM

| `--model` | fits 6 GB? |
|---|---|
| `facebook/mask2former-swin-tiny-coco-instance` | yes, lightest |
| `facebook/mask2former-swin-small-coco-instance` | yes (default) |
| `facebook/mask2former-swin-base-coco-instance` | tight |
| `...-large-...` | likely OOM at inference |

## Fine-tune

Use the HF transformers examples for Mask2Former/MaskFormer instance
segmentation, or the original [`facebookresearch/Mask2Former`](https://github.com/facebookresearch/Mask2Former)
(Detectron2-based) trained on your leaf dataset (CVPPP / Poplar-leaf).

**Related SOTA:** **GMT (Guided Mask Transformer)** builds on this line and
currently tops the CVPPP leaf leaderboard —
[paper](https://arxiv.org/abs/2406.17109) /
[code](https://github.com/vios-s/GMT).

Sources: [Mask2Former paper](https://arxiv.org/abs/2112.01527) ·
[HF model card](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)
