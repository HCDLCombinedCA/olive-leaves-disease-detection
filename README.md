Dataset: https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset
Google drive: https://drive.google.com/drive/folders/1VYd-8RUZ4K9CBlRKwQu6zWmoV34S8oUS

---

## Repository layout

| Path | Contents |
|---|---|
| `Olive_Leaf_CNN_Model_Notebook*.ipynb` | CNN development and saliency maps |
| `glassbox/` | Engineered leaf features and interpretable models |
| `leaf_segmenter/` | SAM-based single-leaf segmentation, and the CVPPP model survey it grew from |
| `compression/` | Model compression (quantisation, pruning) and the edge-efficiency measurements |
| `analysis/` | Corrections and bias investigation, kept separate from the originals |
| `RESULTS.md` | Generated results tables — regenerate with `python3 make_report_tables.py` |
| `draft.md` | Narrative write-up of the experimental work, for lifting into the report |

`compression/` and `analysis/` both run on the unmodified course image
(`kquille/hcaim_gpu_tudublin_course:latest`) and install nothing; see their own
READMEs for what each technique draws on from the labs.

`analysis/` does not modify anything in `glassbox/`, `leaf_segmenter/` or the
notebooks. Each item there is a parallel implementation, so the originals and the
corrected versions can be compared before anything is merged.
