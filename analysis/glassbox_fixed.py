"""Corrected glass-box pipeline: engineered leaf features + interpretable models.

Adapted from `glassbox/Olive_Leaf_GlassBox.ipynb` (Marvin Chewchut). The original
notebook is left untouched; this is a parallel script so the two can be compared
and the team can decide what to merge.

The feature extraction is reproduced faithfully -- same colour fractions, same
GLCM parameters -- so the numbers here are directly comparable to the original.
Three defects are fixed:

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

Usage:
    ./run.sh python glassbox_fixed.py
"""
import argparse
import collections
import json
import os
import re

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


def load_dataset(root):
    """Load every crop with its label, filename and source-photo group.

    Deliberately no shuffling: the original notebook's manual shuffle is what
    desynchronised `paths` from the labels, and the splitters shuffle anyway.
    """
    rows = []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(root, cls)
        for filename in sorted(os.listdir(folder)):
            if os.path.splitext(filename)[1].lower() not in IMG_EXTS:
                continue
            image = Image.open(os.path.join(folder, filename)).convert("RGB")
            array = np.asarray(image.resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.float32) / 255.0
            rows.append({"file": filename, "class": cls, "label": label,
                         "group": source_image(filename), "image": array})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=SEGMENTED_ROOT)
    parser.add_argument("--test-fraction", type=float, default=1 / 3,
                        help="held-out fraction; StratifiedGroupKFold uses 1/n_splits")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="decision tree depth; kept shallow so the tree stays readable")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    print("Loading crops from %s ..." % args.root)
    rows = load_dataset(args.root)
    print("  %d crops from %d source photographs"
          % (len(rows), len({r["group"] for r in rows})))

    multi = collections.Counter(r["group"] for r in rows)
    shared = sum(v for v in multi.values() if v > 1)
    print("  %d crops (%.1f%%) come from photographs that yielded more than one leaf"
          % (shared, shared / len(rows) * 100))

    print("Extracting features ...")
    feature_rows = [leaf_features(r["image"]) for r in rows]
    names = list(feature_rows[0])
    X = np.array([[fr[n] for n in names] for fr in feature_rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])

    os.makedirs(args.out, exist_ok=True)
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "file", [r["file"] for r in rows])
    frame.insert(1, "class", [r["class"] for r in rows])
    frame.insert(2, "label", y)
    frame.insert(3, "source_image", groups)
    csv_path = os.path.join(args.out, "leaf_features_fixed.csv")
    frame.to_csv(csv_path, index=False)
    print("  wrote %s  (%d rows, %d features)" % (csv_path, len(frame), len(names)))

    # Sanity check the fix: every filename must sit in the folder its label names.
    mismatches = sum(
        1 for r in rows
        if not os.path.exists(os.path.join(args.root, r["class"], r["file"])))
    print("  filename/label alignment: %d mismatches (must be 0)" % mismatches)

    n_splits = max(2, int(round(1 / args.test_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    overlap = set(groups[train_idx]) & set(groups[test_idx])
    print("\nSplit: %d train / %d test, source photographs shared between them: %d (must be 0)"
          % (len(train_idx), len(test_idx), len(overlap)))

    results = {}

    tree = DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced",
                                  random_state=SEED).fit(X_train, y_train)
    report("decision_tree", y_test, tree.predict(X_test), results)

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    ).fit(X_train, y_train)
    report("logistic_regression", y_test, logreg.predict(X_test), results)

    # Glass-box explanations: the rubric asks for feature weights and decision paths.
    coefficients = pd.DataFrame(logreg[-1].coef_, columns=names, index=CLASSES).round(3)
    print("\nLogistic regression coefficients (standardised features):")
    print(coefficients.to_string())

    tree_rules = export_text(tree, feature_names=names)
    print("\nDecision tree:")
    print(tree_rules)

    results["logistic_regression_coefficients"] = coefficients.to_dict()
    results["decision_tree_rules"] = tree_rules
    results["split"] = {
        "train": int(len(train_idx)), "test": int(len(test_idx)),
        "grouped_by": "source photograph", "shared_groups": len(overlap),
        "n_splits": n_splits,
    }
    results["features"] = names

    with open(os.path.join(args.out, "glassbox_fixed.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved -> %s" % os.path.join(args.out, "glassbox_fixed.json"))


if __name__ == "__main__":
    main()
