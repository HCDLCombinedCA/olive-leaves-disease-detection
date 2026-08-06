"""Corrected glass-box pipeline: engineered leaf features + interpretable models.

Adapted from `glassbox/Olive_Leaf_GlassBox.ipynb` (Marvin Chewchut). The original
notebook is left untouched; this is a parallel script so the two can be compared
and the team can decide what to merge.

The feature extraction is reproduced faithfully -- same colour fractions, same
GLCM parameters -- so the numbers here are directly comparable to the original.
Four defects are fixed:

1. **Filename/label misalignment in the exported CSV.**
   The notebook shuffles `X` and `Y` but not `paths`, then builds the dataframe
   from the unshuffled `paths` and the shuffled labels. 269 of the first 400 rows
   (67%) carry the wrong filename: `A163_leaf00.png` is listed as
   `olive_peacock_spot` when it comes from the `Healthy/` folder. Model training
   is unaffected -- features and labels stay consistently paired -- but the
   exported `leaf_features.csv` is unusable for tracing a prediction back to an
   image. Fixed here by not shuffling at all: `train_test_split` and
   `StratifiedGroupKFold` both shuffle internally, so the manual shuffle served
   no purpose.

2. **Group leakage across the train/test boundary.**
   The segmenter emits one crop per detected leaf, so a single photograph can
   yield `_leaf00`, `_leaf01`, ... Of 3,003 crops, 550 (18.3%) come from the 267
   photographs that produced more than one. A plain `train_test_split` scatters
   those siblings across both sides, and near-identical crops of the same leaf on
   both sides inflate the score. Fixed with `StratifiedGroupKFold` grouped on the
   source image stem, which also restores the stratification the original call
   omitted.

3. **Only one classical model.**
   The assignment requires two models from its fixed list compared against the
   deep model. Logistic Regression is added alongside the Decision Tree, and its
   coefficients are reported -- the per-class feature weights the rubric asks for
   as the glass-box explanation.

4. **No test set in common with the deep models (`--protocol common`).**
   The segmented crops are all derived from *training* photographs, while the
   CNNs are scored on the official 680-image test split. Scores from the two
   halves therefore answer RQ1 only loosely: they are computed on different
   images. `--protocol common` re-runs the identical feature extraction and
   models over exactly the manifests the CNNs use -- same leakage-free train
   split, same official test images -- so the classical and deep numbers become
   a like-for-like comparison. Both protocols are kept, because the segmented
   one is the setting the features were designed for and the drop between them
   is itself a finding.

Beyond the aggregate scores, `--protocol` runs also emit per-prediction
explanations: two correct and two incorrect test cases per model, each with the
decision path (tree) or the signed per-feature contribution (logistic
regression) that produced the prediction.

Usage:
    ./run.sh python glassbox_fixed.py                      # segmented crops
    ./run.sh python glassbox_fixed.py --protocol common    # official test split
"""
import argparse
import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
SEGMENTED_ROOT = "../olive_leaf_dataset/segmented/train"
PREPARED_ROOT = "../compression/data/prepared"
IMAGE_SIZE = (224, 224)
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def leaf_features(img):
    """Colour composition and texture descriptors for one leaf crop.

    Reproduced from the original notebook so results stay comparable. Features
    that were written but commented out there (lesion morphology, LBP histogram,
    Sobel edges) are likewise left out.

    Note for the report: the GLCM is computed over the whole crop, not restricted
    to the leaf mask, so it also describes background texture. The colour
    fractions *are* masked. This matters given how strongly capture source
    correlates with class in this dataset (see acquisition_bias.py).
    """
    img = np.asarray(img, dtype=float)
    if img.max() > 1.5:
        img = img / 255.0
    hsv = rgb2hsv(img)
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    gray = rgb2gray(img)
    features = {}

    # Leaf mask: drop near-white and near-black background.
    leaf = (val > 0.15) & ~((sat < 0.15) & (val > 0.85))
    if leaf.sum() < 100:
        leaf = np.ones_like(val, dtype=bool)
    n_leaf = leaf.sum()

    def frac(mask):
        return (mask & leaf).sum() / n_leaf

    features["frac_green"] = frac((hue >= 70) & (hue < 160) & (sat > 0.25))
    features["frac_yellow"] = frac((hue >= 40) & (hue < 70) & (sat > 0.25))   # chlorosis
    features["frac_brown"] = frac((hue >= 10) & (hue < 40) & (sat > 0.20) & (val < 0.6))
    features["frac_dark"] = frac(val < 0.25)                                  # necrosis
    features["frac_gray"] = frac(sat < 0.20)                                  # silvering
    quantised = (gray * 63).astype(np.uint8)
    glcm = graycomatrix(quantised, distances=[1, 3], angles=[0, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        features["glcm_%s" % prop] = float(graycoprops(glcm, prop).mean())
    return features


def source_image(filename):
    """Strip the per-leaf suffix to recover the photograph a crop came from.

    `a231_leaf00.png` and `a231_leaf01.png` are two leaves from one photo and
    must never straddle the train/test split.
    """
    return re.sub(r"_leaf\d+\.(png|jpg|jpeg)$", "", filename, flags=re.I)


def read_image(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.float32) / 255.0


def load_segmented(root):
    """Every segmented crop with its label, path and source-photo group.

    Deliberately no shuffling: the original notebook's manual shuffle is what
    desynchronised `paths` from the labels, and the splitters shuffle anyway.
    Features are extracted as the images are read rather than afterwards, so
    only one decoded image is held in memory at a time.
    """
    rows = []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(root, cls)
        for filename in sorted(os.listdir(folder)):
            if os.path.splitext(filename)[1].lower() not in IMG_EXTS:
                continue
            path = os.path.join(folder, filename)
            rows.append({"file": filename, "path": path, "class": cls, "label": label,
                         "group": source_image(filename), "split": None,
                         "features": leaf_features(read_image(path))})
    return rows


def load_prepared(root):
    """The exact images the CNNs are trained and scored on.

    Reads `compression/src/prepare_data.py`'s manifests, so the train/val/test
    membership here is identical to the deep models' -- same de-duplication,
    same group-aware split, same official test images. `split` is carried
    through and used instead of a new random division.
    """
    images_dir = os.path.join(root, "images")
    frames = [
        pd.read_csv(os.path.join(root, "manifest_trainval.csv")),
        pd.read_csv(os.path.join(root, "manifest_test.csv")).assign(split="test"),
    ]
    rows = []
    for frame in frames:
        for record in frame.to_dict("records"):
            path = os.path.join(images_dir, record["file"])
            rows.append({
                "file": record["file"], "path": path, "class": record["class"],
                "label": int(record["label"]),
                # Whole photographs, one leaf each: the photograph is its own group.
                "group": record["file"], "split": record["split"],
                "features": leaf_features(read_image(path)),
            })
    return rows


def report(name, y_true, y_pred, store):
    print("\n--- %s ---" % name)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("macro-F1: %.4f" % f1_score(y_true, y_pred, average="macro"))
    store[name] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
        "report": classification_report(y_true, y_pred, target_names=CLASSES,
                                        digits=3, output_dict=True),
    }


def choose_examples(y_true, y_pred, probs, n_each=2):
    """Pick two correct and two incorrect test predictions to explain.

    Selection is deterministic and deliberately not random: the most confident
    cases are the ones whose explanation is worth reading. Confident correct
    predictions show which features the model actually relies on; confident
    *errors* are where a glass-box model earns its keep, because the reason for
    the mistake is legible rather than hidden. Classes are spread out where
    possible so the four cases do not all describe the same disease.
    """
    confidence = probs.max(axis=1)
    chosen = []
    for correct in (True, False):
        pool = [i for i in range(len(y_true)) if bool(y_true[i] == y_pred[i]) == correct]
        pool.sort(key=lambda i: -confidence[i])
        seen, picked = set(), []
        for index in pool:                      # first pass: one per true class
            if y_true[index] not in seen:
                picked.append(index)
                seen.add(y_true[index])
            if len(picked) == n_each:
                break
        for index in pool:                      # top up if a class was unavailable
            if len(picked) == n_each:
                break
            if index not in picked:
                picked.append(index)
        chosen.extend(picked)
    return chosen


def explain_tree(tree, x, names):
    """The decision path taken by one sample, as readable comparisons.

    A tree's explanation is not a weighting but a sequence of tests, so it is
    reported as the tests themselves: the feature, the threshold, and the value
    that sent this sample left or right.
    """
    path = tree.decision_path(x.reshape(1, -1)).indices
    left, right = tree.tree_.children_left, tree.tree_.children_right
    steps = []
    for node in path:
        if left[node] == right[node]:           # leaf
            counts = tree.tree_.value[node][0]
            steps.append({"leaf": True,
                          "class_distribution": {c: float(v) for c, v in zip(CLASSES, counts)}})
            continue
        feature = names[tree.tree_.feature[node]]
        threshold = float(tree.tree_.threshold[node])
        value = float(x[tree.tree_.feature[node]])
        steps.append({
            "leaf": False, "feature": feature, "threshold": round(threshold, 4),
            "value": round(value, 4),
            "test": "%s = %.4f %s %.4f" % (feature, value,
                                           "<=" if value <= threshold else ">", threshold),
        })
    return steps


def explain_logreg(pipeline, x, names, target):
    """Signed per-feature contributions to one class's decision score.

    For a linear model the score is an exact sum, so the contribution of each
    feature is `coefficient x standardised value` with no approximation
    involved -- unlike the LIME surrogates used for the deep models. Reported
    against the standardised features because the raw ones differ by orders of
    magnitude and their coefficients would not be comparable.
    """
    scaler, model = pipeline[0], pipeline[-1]
    z = scaler.transform(x.reshape(1, -1))[0]
    contributions = model.coef_[target] * z
    order = np.argsort(-np.abs(contributions))
    return {
        "intercept": float(model.intercept_[target]),
        "decision_score": float(contributions.sum() + model.intercept_[target]),
        "contributions": [
            {"feature": names[i], "raw_value": round(float(x[i]), 4),
             "standardised_value": round(float(z[i]), 3),
             "coefficient": round(float(model.coef_[target][i]), 3),
             "contribution": round(float(contributions[i]), 3)}
            for i in order
        ],
    }


def explanation_figure(model_name, cases, path, axis_label):
    """One panel per explained case: the image beside its feature contributions.

    Kept to the four selected cases so the figure stays readable in a six-page
    report; the full numeric detail lives in the JSON alongside it.
    """
    rows = len(cases)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.9 * rows),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    axes = np.atleast_2d(axes)
    for row, case in enumerate(cases):
        image_ax, bar_ax = axes[row]
        image_ax.imshow(read_image(case["path"]))
        image_ax.set_xticks([]), image_ax.set_yticks([])
        verdict = "correct" if case["correct"] else "INCORRECT"
        # Three short lines rather than one long one: the class names are long
        # enough that a single line overflows the image axis and gets clipped.
        image_ax.set_title("%s\ntrue: %s\npred: %s (p=%.2f)  [%s]"
                           % (os.path.basename(case["file"]), case["true_class"],
                              case["predicted_class"], case["confidence"], verdict),
                           fontsize=8)

        bars = case["top_contributions"]
        labels = [b.get("label", b["feature"]) for b in bars][::-1]
        values = [b["contribution"] for b in bars][::-1]
        colours = ["#c0392b" if v < 0 else "#27ae60" for v in values]
        bar_ax.barh(range(len(values)), values, color=colours)
        bar_ax.set_yticks(range(len(values)))
        bar_ax.set_yticklabels(labels, fontsize=8)
        bar_ax.axvline(0, color="black", linewidth=0.8)
        bar_ax.set_xlabel(axis_label.format(predicted=case["predicted_class"]), fontsize=8)
        bar_ax.tick_params(labelsize=8)
    fig.suptitle("%s: two correct and two incorrect test predictions" % model_name,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  wrote %s" % path)


def build_cases(kind, model, X_test, y_test, y_pred, probs, rows, test_idx, names, scale):
    """Assemble the four explained predictions for one model.

    `scale` is the per-feature training standard deviation, used to put the
    decision tree's margins on a comparable axis: the raw distance from a
    threshold is not comparable between `frac_green` (bounded by 1) and
    `glcm_contrast` (unbounded), so plotting the raw differences side by side
    would make the smallest-scale feature look irrelevant.
    """
    cases = []
    for position in choose_examples(y_test, y_pred, probs):
        row = rows[test_idx[position]]
        x = X_test[position]
        predicted = int(y_pred[position])
        case = {
            "file": row["file"], "path": row["path"],
            "true_class": CLASSES[int(y_test[position])],
            "predicted_class": CLASSES[predicted],
            "confidence": float(probs[position].max()),
            "correct": bool(y_test[position] == y_pred[position]),
            "feature_values": {n: round(float(v), 4) for n, v in zip(names, x)},
        }
        if kind == "decision_tree":
            case["decision_path"] = explain_tree(model, x, names)
            # A tree has no per-feature weights. The closest equivalent is how
            # decisively the sample passed each test on its path, so the bars
            # show the distance from each threshold in standard deviations of
            # that feature, and the label states the test that was applied.
            case["top_contributions"] = [
                {"feature": step["feature"],
                 "label": step["test"],
                 "contribution": (step["value"] - step["threshold"])
                                 / scale[names.index(step["feature"])]}
                for step in case["decision_path"] if not step["leaf"]
            ]
        else:
            explanation = explain_logreg(model, x, names, predicted)
            case["linear_explanation"] = explanation
            case["top_contributions"] = [
                {"feature": c["feature"], "contribution": c["contribution"]}
                for c in explanation["contributions"][:6]
            ]
            if not case["correct"]:
                # Why the true class lost: the same decomposition for the class
                # the model should have chosen.
                case["true_class_explanation"] = explain_logreg(
                    model, x, names, int(y_test[position]))
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=("segmented", "common"), default="segmented",
                        help="'segmented' uses the segmenter's crops with a grouped "
                             "hold-out; 'common' uses the same manifests and official "
                             "test split as the CNNs, for a like-for-like RQ1 comparison")
    parser.add_argument("--root", default=None,
                        help="dataset root; defaults to the protocol's usual location")
    parser.add_argument("--test-fraction", type=float, default=1 / 3,
                        help="segmented protocol only; StratifiedGroupKFold uses 1/n_splits")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="decision tree depth; kept shallow so the tree stays readable")
    parser.add_argument("--out", default="results")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    common_protocol = args.protocol == "common"
    root = args.root or (PREPARED_ROOT if common_protocol else SEGMENTED_ROOT)
    stem = args.out_name or ("glassbox_common_protocol" if common_protocol else "glassbox_fixed")

    print("Loading %s images from %s ..." % (args.protocol, root))
    rows = load_prepared(root) if common_protocol else load_segmented(root)
    print("  %d images from %d source photographs"
          % (len(rows), len({r["group"] for r in rows})))

    multi = collections.Counter(r["group"] for r in rows)
    shared = sum(v for v in multi.values() if v > 1)
    print("  %d images (%.1f%%) come from photographs that yielded more than one leaf"
          % (shared, shared / len(rows) * 100))

    names = list(rows[0]["features"])
    X = np.array([[r["features"][n] for n in names] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    os.makedirs(args.out, exist_ok=True)
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "file", [r["file"] for r in rows])
    frame.insert(1, "class", [r["class"] for r in rows])
    frame.insert(2, "label", y)
    frame.insert(3, "source_image", groups)
    csv_path = os.path.join(args.out, "leaf_features_%s.csv"
                            % ("common_protocol" if common_protocol else "fixed"))
    frame.to_csv(csv_path, index=False)
    print("  wrote %s  (%d rows, %d features)" % (csv_path, len(frame), len(names)))

    # Sanity check the fix: every path must still resolve to a real file.
    mismatches = sum(1 for r in rows if not os.path.exists(r["path"]))
    print("  filename/label alignment: %d mismatches (must be 0)" % mismatches)

    if common_protocol:
        # No new split is drawn: membership comes from the manifests, so the
        # classical models see exactly the images the CNNs saw. Validation is
        # folded into training because these models have nothing to early-stop.
        train_idx = np.array([i for i, r in enumerate(rows) if r["split"] in ("train", "val")])
        test_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
        split_note = "official test split from compression/data/prepared manifests"
        n_splits = None
    else:
        n_splits = max(2, int(round(1 / args.test_fraction)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        split_note = "StratifiedGroupKFold on the source photograph"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    overlap = set(groups[train_idx]) & set(groups[test_idx])
    print("\nSplit: %d train / %d test (%s), source photographs shared between them: %d (must be 0)"
          % (len(train_idx), len(test_idx), split_note, len(overlap)))

    results = {}
    explanations = {}

    tree = DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced",
                                  random_state=SEED).fit(X_train, y_train)
    tree_probs = tree.predict_proba(X_test)
    report("decision_tree", y_test, tree.predict(X_test), results)

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    ).fit(X_train, y_train)
    logreg_probs = logreg.predict_proba(X_test)
    report("logistic_regression", y_test, logreg.predict(X_test), results)

    # Glass-box explanations: the rubric asks for feature weights and decision paths.
    coefficients = pd.DataFrame(logreg[-1].coef_, columns=names, index=CLASSES).round(3)
    print("\nLogistic regression coefficients (standardised features):")
    print(coefficients.to_string())

    tree_rules = export_text(tree, feature_names=names)
    print("\nDecision tree:")
    print(tree_rules)

    # ---- Individual predictions, which the global views above cannot show ----
    print("\nExplaining individual predictions (2 correct, 2 incorrect per model) ...")
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    axis_labels = {
        "decision_tree": "distance from each threshold on the path (SD of the feature)",
        "logistic_regression": "contribution to the score for '{predicted}'",
    }
    for kind, model, probs in (("decision_tree", tree, tree_probs),
                               ("logistic_regression", logreg, logreg_probs)):
        cases = build_cases(kind, model, X_test, y_test, probs.argmax(axis=1),
                            probs, rows, test_idx, names, scale)
        explanations[kind] = cases
        for case in cases:
            print("  [%s] %-28s true=%-18s pred=%-18s p=%.2f %s"
                  % (kind[:4], case["file"][-28:], case["true_class"],
                     case["predicted_class"], case["confidence"],
                     "" if case["correct"] else "<- error"))
        explanation_figure(kind.replace("_", " "), cases,
                           os.path.join(args.out, "%s_%s_examples.png" % (stem, kind)),
                           axis_labels[kind])

    results["logistic_regression_coefficients"] = coefficients.to_dict()
    results["decision_tree_rules"] = tree_rules
    results["explanations"] = explanations
    results["split"] = {
        "protocol": args.protocol,
        "train": int(len(train_idx)), "test": int(len(test_idx)),
        "grouped_by": "source photograph", "shared_groups": len(overlap),
        "n_splits": n_splits, "note": split_note,
    }
    results["features"] = names

    with open(os.path.join(args.out, "%s.json" % stem), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved -> %s" % os.path.join(args.out, "%s.json" % stem))


if __name__ == "__main__":
    main()
