"""Does capture source predict the disease label?

`review.txt` section 3.6.3 flagged background/camera bias as a risk and nobody
had tested it. This script tests it two ways.

**Part A -- the confound in the data.** Filenames carry the capture device:
`B-*` and `DSC_*` are cameras, `IMG_YYYYMMDD_HHMMSS` is a phone, and one block is
bare numbers. Cross-tabulating prefix against class shows how far the two are
entangled, and a classifier given *nothing but the filename prefix* puts a number
on it. If that classifier scores well, then any image model trained here can
reach a similar score without looking at a single leaf, and reported accuracy
cannot be read as evidence of disease recognition.

**Part B -- where the model actually looks.** Grad-CAM++ attention is measured
inside versus outside a leaf mask. The mask reuses the HSV rule from the team's
own `glassbox` feature extractor, so the two components agree on what counts as
leaf. A model attending mostly to background is relying on the confound from
Part A; a model attending to the leaf is not.

Part B needs a trained model and is skipped with `--skip-gradcam` if none exists.

Usage:
    ./run.sh python acquisition_bias.py
    ./run.sh python acquisition_bias.py --model ../compression/artifacts/mobilenetv2_baseline
"""
import argparse
import collections
import json
import os
import re
import zipfile

import numpy as np

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
SEED = 42


def capture_source(filename):
    """Group a filename by the device that produced it."""
    lower = filename.lower()
    if lower.startswith("img_"):
        return "IMG_ (phone, timestamped)"
    if lower.startswith("dsc_"):
        return "DSC_ (DSLR)"
    if lower.startswith("b"):
        return "B* (camera roll)"
    if re.match(r"^a[-\d]", lower):
        return "A* (camera roll)"
    if re.match(r"^\d", lower):
        return "bare numeric"
    return "other"


def read_index(zip_path):
    """(split, class, filename) for every image, without decoding pixels."""
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            parts = member.split("/")
            if member.endswith("/") or len(parts) < 4 or parts[2] not in CLASSES:
                continue
            rows.append((parts[1], parts[2], parts[3]))
    return rows


def part_a(rows):
    """Cross-tabulate capture source against class, then predict class from source."""
    print("=" * 78)
    print("PART A -- is capture source entangled with the label?")
    print("=" * 78)

    sources = sorted({capture_source(f) for _s, _c, f in rows})
    table = collections.defaultdict(collections.Counter)
    for split, cls, filename in rows:
        table[(split, cls)][capture_source(filename)] += 1

    header = "%-26s" % "split/class" + "".join("%>22s".replace(">", "") % s[:20] for s in sources)
    print("\n" + header)
    for key in sorted(table):
        total = sum(table[key].values())
        line = "%-26s" % ("%s/%s" % key)
        for source in sources:
            count = table[key][source]
            line += "%10d (%4.0f%%)" % (count, count / total * 100)
        print(line)

    # A classifier that sees only the capture source. Predicting the most common
    # class for each source is the strongest such rule, so its accuracy is an
    # upper bound on how much of the label a source-only model can recover.
    train = [(capture_source(f), c) for s, c, f in rows if s == "train"]
    test = [(capture_source(f), c) for s, c, f in rows if s == "test"]

    majority = {}
    by_source = collections.defaultdict(collections.Counter)
    for source, cls in train:
        by_source[source][cls] += 1
    for source, counts in by_source.items():
        majority[source] = counts.most_common(1)[0][0]
    fallback = collections.Counter(c for _s, c in train).most_common(1)[0][0]

    def score(pairs):
        correct = sum(1 for source, cls in pairs if majority.get(source, fallback) == cls)
        return correct / len(pairs)

    baseline = max(collections.Counter(c for _s, c in test).values()) / len(test)
    print("\nClass purity of each capture source in the training split:")
    for source, counts in sorted(by_source.items()):
        total = sum(counts.values())
        top_cls, top_n = counts.most_common(1)[0]
        print("  %-26s n=%5d  ->  %-20s %5.1f%%" % (source, total, top_cls, top_n / total * 100))

    print("\nPredicting the class from the filename prefix alone:")
    print("  train accuracy            %.3f" % score(train))
    print("  test  accuracy            %.3f" % score(test))
    print("  majority-class baseline   %.3f" % baseline)
    print("\n  A model that never sees a leaf reaches %.1f%% on the official test split."
          % (score(test) * 100))

    return {
        "source_class_table": {"%s/%s" % k: dict(v) for k, v in table.items()},
        "source_purity": {s: {"n": sum(c.values()), "top_class": c.most_common(1)[0][0],
                              "purity": c.most_common(1)[0][1] / sum(c.values())}
                          for s, c in by_source.items()},
        "filename_only_accuracy": {"train": score(train), "test": score(test),
                                   "majority_baseline": baseline},
    }


def flatten_model(model):
    """Rebuild a Sequential([backbone, head...]) as one flat functional model.

    tf-keras-vis walks `model.layers` to find the last convolutional layer. When
    the backbone is nested inside a Sequential it only sees three layers -- the
    backbone as a single opaque block, then the head -- and rebuilding the graph
    from the outer input fails with "Graph disconnected". Re-applying the head to
    the backbone's output exposes every convolution at the top level, which is
    what Grad-CAM needs.
    """
    import tensorflow as tf

    if not isinstance(model.layers[0], tf.keras.Model):
        return model
    backbone = model.layers[0]
    x = backbone.output
    for layer in model.layers[1:]:
        x = layer(x)
    return tf.keras.Model(inputs=backbone.input, outputs=x)


def leaf_mask(image01):
    """The HSV leaf rule used by the team's glassbox feature extractor."""
    from skimage.color import rgb2hsv

    hsv = rgb2hsv(image01)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = (val > 0.15) & ~((sat < 0.15) & (val > 0.85))
    if mask.sum() < 100:
        mask = np.ones_like(val, dtype=bool)
    return mask


def part_b(model_dir, manifest, images_dir, per_class, out):
    """Fraction of Grad-CAM++ attention landing outside the leaf."""
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore

    print("\n" + "=" * 78)
    print("PART B -- does the model attend to the leaf or to the background?")
    print("=" * 78)

    model = flatten_model(tf.keras.models.load_model(model_dir))
    # Preprocessing has to match how the model was trained.
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input \
        if "mobilenet" in model_dir.lower() \
        else tf.keras.applications.densenet.preprocess_input

    frame = pd.read_csv(manifest)
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)

    records = []
    for cls_index, cls in enumerate(CLASSES):
        # Random sample, not the first N alphabetically: filenames are ordered by
        # capture source, so `.head()` would draw every Healthy image from the
        # B-* camera and every peacock image from the DSLR -- exactly the
        # confound under investigation.
        subset = frame[frame["class"] == cls].sample(
            n=min(per_class, (frame["class"] == cls).sum()), random_state=SEED)
        for _, row in subset.iterrows():
            raw = img_to_array(load_img(os.path.join(images_dir, row["file"]),
                                        target_size=(224, 224))) / 255.0
            batch = preprocess(raw[np.newaxis, ...] * 255.0)
            probs = model.predict(batch, verbose=0)[0]
            predicted = int(probs.argmax())

            cam = gradcam(CategoricalScore([predicted]), batch, penultimate_layer=-1)[0]
            cam = np.asarray(cam, dtype=np.float64)
            if cam.max() > 0:
                cam = cam / cam.max()

            mask = leaf_mask(raw)
            total = cam.sum()
            outside = float(cam[~mask].sum() / total) if total > 0 else float("nan")
            records.append({
                "file": row["file"], "true_class": cls,
                "predicted": CLASSES[predicted], "correct": predicted == cls_index,
                "confidence": float(probs[predicted]),
                "leaf_area_fraction": float(mask.mean()),
                "attention_outside_leaf": outside,
            })

    print("\n%-34s %-18s %6s %12s %14s" % (
        "file", "pred", "conf", "leaf area", "attn outside"))
    print("-" * 92)
    for record in records:
        print("%-34s %-18s %6.3f %11.1f%% %13.1f%%" % (
            record["file"].split("/")[-1][:33], record["predicted"][:17],
            record["confidence"], record["leaf_area_fraction"] * 100,
            record["attention_outside_leaf"] * 100))

    outside = np.array([r["attention_outside_leaf"] for r in records])
    leaf_area = np.array([r["leaf_area_fraction"] for r in records])
    correct = np.array([r["correct"] for r in records])
    # If attention were spread evenly, the share landing outside the leaf would
    # equal the background's share of the image. Comparing against that tells us
    # whether the model prefers background or merely encounters it.
    expected = 1 - leaf_area
    def summarise(label, selection):
        if selection.sum() == 0:
            print("  %-26s (none)" % label)
            return None
        ratio = outside[selection].mean() / expected[selection].mean()
        print("  %-26s n=%3d  attn outside %5.1f%%  background %5.1f%%  ratio %.2f"
              % (label, selection.sum(), outside[selection].mean() * 100,
                 expected[selection].mean() * 100, ratio))
        return float(ratio)

    print("\n  A ratio above 1 means the model puts more attention on background")
    print("  than an even spread would, i.e. it is leaning on the confound.\n")
    ratio_all = summarise("all images", np.ones(len(records), dtype=bool))
    ratio_ok = summarise("correct predictions", correct)
    ratio_bad = summarise("incorrect predictions", ~correct)

    return {
        "model": model_dir,
        "per_image": records,
        "mean_attention_outside_leaf": float(outside.mean()),
        "mean_background_area": float(expected.mean()),
        "background_preference_ratio": ratio_all,
        "background_preference_ratio_correct": ratio_ok,
        "background_preference_ratio_incorrect": ratio_bad,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="../olive-leaf-image-dataset.zip")
    parser.add_argument("--model", default="../compression/artifacts/mobilenetv2_baseline")
    parser.add_argument("--manifest", default="../compression/data/prepared/manifest_test.csv")
    parser.add_argument("--images", default="../compression/data/prepared/images")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    results = {"part_a": part_a(read_index(args.zip))}

    if args.skip_gradcam:
        print("\n(Part B skipped)")
    elif not os.path.isdir(args.model):
        print("\n(Part B skipped: no model at %s)" % args.model)
    else:
        results["part_b"] = part_b(args.model, args.manifest, args.images,
                                   args.per_class, args.out)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "acquisition_bias.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
