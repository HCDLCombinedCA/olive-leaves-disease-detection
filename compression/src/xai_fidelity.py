"""Compare explanations before and after compression, using LIME.

This answers the second half of RQ2 -- whether compression preserves *explanation
fidelity*, not just accuracy -- and simultaneously supplies the second XAI method
the project is missing (the existing notebook only uses Grad-CAM++).

Why LIME rather than Grad-CAM
-----------------------------
Grad-CAM needs gradients, and a `.tflite` model has none: it is an inference-only
graph. LIME is model-agnostic -- it only needs a function from images to class
probabilities -- so exactly the same explanation procedure runs against the Keras
baseline and against every quantised variant. That makes the before/after
comparison a like-for-like one. LIME is also what the course covers in
Week 8 (`Lime and Shap.ipynb`) and ships in the course image.

Making the comparison fair
--------------------------
Two controls matter, and without them the "difference" measured is mostly noise:

1. **The same segmentation** is computed once per image and handed to both
   explanations, so superpixel *k* means the same region in both.
2. **The same random seed**, so both models see the identical set of perturbed
   samples. Any remaining difference in the fitted weights is then attributable
   to the models, not the sampling.

Metrics
-------
* **Spearman rank correlation** between the two per-superpixel weight vectors:
  does the compressed model rank the evidence in the same order?
* **Top-k Jaccard overlap**: of the k regions each model leans on most, how many
  are shared? This is the more intuitive number for the report -- "the quantised
  model relies on the same 4 of the top 5 regions".
* **Agreement of the predicted label**, reported alongside, since an explanation
  comparison is only meaningful when both models are answering the same way.

Usage:
    ./run.sh python src/xai_fidelity.py --backbone mobilenetv2
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from skimage.segmentation import slic

import common
from quantise import tflite_predictor

# Fixed SLIC segmentation. SLIC is used rather than LIME's default quickshift
# because it takes an explicit segment count, which keeps the superpixel grid
# comparable across images.
SLIC_SEGMENTS = 60
SLIC_COMPACTNESS = 10.0
TOP_K = 5


def load_raw_images(manifest_path, images_dir, per_class, size):
    """Load a few images per class in raw RGB [0, 1] -- LIME perturbs this space."""
    import pandas as pd
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    frame = pd.read_csv(manifest_path)
    picked = []
    for cls in common.CLASSES:
        rows = frame[frame["class"] == cls].head(per_class)
        for _, row in rows.iterrows():
            path = os.path.join(images_dir, row["file"])
            array = img_to_array(load_img(path, target_size=size)) / 255.0
            picked.append({"file": row["file"], "class": cls,
                           "label": common.CLASSES.index(cls), "image": array})
    return picked


def make_classifier_fn(predict, preprocess_fn):
    """Adapt a model to LIME's interface.

    LIME hands over a batch of images in the same space as the one it was given
    (here RGB in [0, 1]); preprocessing has to happen inside this wrapper so that
    both the Keras and tf.lite paths see exactly the input they were trained on.
    """
    def classifier_fn(images):
        batch = preprocess_fn(np.asarray(images, dtype=np.float32) * 255.0)
        return predict(batch)
    return classifier_fn


def weight_vector(explanation, label, num_segments):
    """Dense per-superpixel weight vector from a LIME explanation."""
    weights = np.zeros(num_segments, dtype=np.float64)
    for segment_id, weight in explanation.local_exp.get(label, []):
        if segment_id < num_segments:
            weights[segment_id] = weight
    return weights


def compare(a, b, top_k=TOP_K):
    """Fidelity metrics between two per-superpixel weight vectors."""
    if np.allclose(a, 0) or np.allclose(b, 0):
        rho = float("nan")
    else:
        rho = float(spearmanr(a, b).correlation)
    top_a = set(np.argsort(-np.abs(a))[:top_k])
    top_b = set(np.argsort(-np.abs(b))[:top_k])
    union = top_a | top_b
    return {
        "spearman": rho,
        "top%d_jaccard" % top_k: len(top_a & top_b) / len(union) if union else float("nan"),
        "top%d_shared" % top_k: len(top_a & top_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--variants", nargs="+",
                        default=["dynamic_range", "float16", "full_integer"],
                        help="tflite variants to compare against the Keras baseline")
    parser.add_argument("--per-class", type=int, default=2, help="images per class")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="LIME perturbation samples per explanation")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    from lime import lime_image

    common.set_seeds(args.seed)
    _constructor, preprocess_fn, _prefix = common.BACKBONES[args.backbone]

    baseline_dir = os.path.join(args.artifacts, "%s_baseline" % args.backbone)
    if not os.path.isdir(baseline_dir):
        raise SystemExit("baseline not found: %s" % baseline_dir)
    baseline = tf.keras.models.load_model(baseline_dir)
    baseline_fn = make_classifier_fn(
        lambda batch: baseline.predict(batch, verbose=0), preprocess_fn)

    compressed = {}
    for variant in args.variants:
        path = os.path.join(args.artifacts,
                            "%s_baseline_%s.tflite" % (args.backbone, variant))
        if not os.path.isfile(path):
            print("skipping %s: %s not found" % (variant, path))
            continue
        compressed[variant] = make_classifier_fn(tflite_predictor(path), preprocess_fn)

    samples = load_raw_images(common.TEST_MANIFEST, common.IMAGES_DIR,
                              args.per_class, common.IMAGE_SIZE)
    explainer = lime_image.LimeImageExplainer(random_state=args.seed)

    records = []
    for sample in samples:
        image = sample["image"]
        segments = slic(image, n_segments=SLIC_SEGMENTS,
                        compactness=SLIC_COMPACTNESS, start_label=0)
        num_segments = int(segments.max()) + 1
        segmentation_fn = lambda _img: segments      # identical regions for every model

        base_probs = baseline_fn([image])[0]
        base_label = int(base_probs.argmax())

        base_explanation = explainer.explain_instance(
            image.astype(np.double), baseline_fn, labels=(base_label,),
            top_labels=None, num_samples=args.num_samples,
            segmentation_fn=segmentation_fn, random_seed=args.seed)
        base_weights = weight_vector(base_explanation, base_label, num_segments)

        print("\n%s  true=%s  baseline=%s (%.3f)%s"
              % (sample["file"], sample["class"], common.CLASSES[base_label],
                 base_probs[base_label],
                 "" if base_label == sample["label"] else "   <- misclassified"))

        for variant, classifier_fn in compressed.items():
            variant_probs = classifier_fn([image])[0]
            variant_label = int(variant_probs.argmax())
            variant_explanation = explainer.explain_instance(
                image.astype(np.double), classifier_fn, labels=(base_label,),
                top_labels=None, num_samples=args.num_samples,
                segmentation_fn=segmentation_fn, random_seed=args.seed)
            variant_weights = weight_vector(variant_explanation, base_label, num_segments)

            metrics = compare(base_weights, variant_weights)
            metrics.update({
                "file": sample["file"],
                "true_class": sample["class"],
                "baseline_prediction": common.CLASSES[base_label],
                "baseline_correct": base_label == sample["label"],
                "variant": variant,
                "variant_prediction": common.CLASSES[variant_label],
                "label_agreement": variant_label == base_label,
                "segments": num_segments,
            })
            records.append(metrics)
            print("    %-14s label=%-19s rho=%+.3f  top%d shared=%d/%d"
                  % (variant, common.CLASSES[variant_label], metrics["spearman"],
                     TOP_K, metrics["top%d_shared" % TOP_K], TOP_K))

    summary = {}
    for variant in compressed:
        subset = [r for r in records if r["variant"] == variant]
        if not subset:
            continue
        rhos = [r["spearman"] for r in subset if not np.isnan(r["spearman"])]
        summary[variant] = {
            "images": len(subset),
            "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
            "mean_top%d_jaccard" % TOP_K: round(float(np.mean(
                [r["top%d_jaccard" % TOP_K] for r in subset])), 4),
            "label_agreement_rate": round(float(np.mean(
                [r["label_agreement"] for r in subset])), 4),
        }

    common.save_json({"per_image": records, "summary": summary,
                      "settings": {"num_samples": args.num_samples,
                                   "slic_segments": SLIC_SEGMENTS, "top_k": TOP_K,
                                   "seed": args.seed}},
                     os.path.join("results", "%s_xai_fidelity.json" % args.backbone))

    print("\n%-16s %8s %14s %18s" % ("variant", "images", "mean Spearman", "mean top-5 Jaccard"))
    print("-" * 60)
    for variant, entry in summary.items():
        print("%-16s %8d %14s %18.3f" % (
            variant, entry["images"],
            "n/a" if entry["mean_spearman"] is None else "%.3f" % entry["mean_spearman"],
            entry["mean_top%d_jaccard" % TOP_K]))
    print("\nsaved -> results/%s_xai_fidelity.json" % args.backbone)


if __name__ == "__main__":
    main()
