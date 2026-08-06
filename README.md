Dataset: https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset
Google drive: https://drive.google.com/drive/folders/1VYd-8RUZ4K9CBlRKwQu6zWmoV34S8oUS

---

## Repository layout

| Path | Contents |
|---|---|
| `Olive_Leaf_Final_Integrated_Notebook.ipynb` | Executed integration notebook: segmentation, from-scratch and transfer-learned CNNs, glass-box models, compression, XAI, and bias analysis in one runnable document |
| `Olive_Leaf_CNN_Model_Notebook*.ipynb` | CNN development and saliency maps |
| `glassbox/` | Engineered leaf features and interpretable models |
| `leaf_segmenter/` | Single-leaf segmentation (fine-tuned YOLO11-seg) and the CVPPP model survey it was chosen from |
| `compression/` | Model training (`train_baseline.py`, `train_scratch.py`), compression, and the edge-efficiency measurements |
| `analysis/` | Corrections and bias investigation, kept separate from the originals |
| `RESULTS.md` | Generated results tables — regenerate with `python3 make_report_tables.py` |
| `draft*.md` | Narrative write-ups (experiments, EDA, XAI, literature review, ethics) for lifting into the report |

`compression/` and `analysis/` both run on the unmodified course image
(`kquille/hcaim_gpu_tudublin_course:latest`) and install nothing; see their own
READMEs for what each technique draws on from the labs.

`analysis/` does not modify anything in `glassbox/`, `leaf_segmenter/` or the
notebooks. Each item there is a parallel implementation, so the originals and the
corrected versions can be compared before anything is merged.

Trained models (`compression/artifacts/`, 394 MB) and the prepared image copies
(`compression/data/prepared/images/`, 95 MB) are not in version control. The
manifests and split statistics beside them are, so a fresh clone can still see
exactly which images were dropped as leaked and reproduce the split. The
integration notebook reports which stages the working copy holds in its first
cell, and skips the rest with an explanation rather than failing.
