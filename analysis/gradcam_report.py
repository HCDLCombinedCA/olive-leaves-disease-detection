"""Grad-CAM++ explanations on a model that actually works.

Replaces the saliency section of `Olive_Leaf_CNN_Model_Notebook_withSaliency.ipynb`
(NicholusMagak). That notebook is left untouched; this produces a standalone
figure set the team can merge.

What was wrong with the original
--------------------------------
1. **The explained model is the broken one.** Cells 31-32 save `modelBest`, the
   scratch CNN, and cell 35 loads it back, so every heatmap in cells 39-45
   describes a model with macro-F1 0.401 whose recall on `aculus_olearius` is
   0.5% -- it essentially never predicts that class. Confirmed by reading the
   architecture out of the committed `1785117388.zip`. The DenseNet121 that
   reached 0.946 was never saved at all.
2. **Heatmaps are generated for the true label, not the predicted one.** For a
   model that never predicts a class, asking "where is the evidence for that
   class" produces a map of a pathway that does not fire. For error analysis you
   want both: what drove the prediction the model *made*, and what it found for
   the label it *should* have chosen.
3. **Only correct predictions are shown**, three per class, taken as the first
   three filenames alphabetically. The rubric asks for at least two correct and
   two incorrect per class.
4. **No labels on the figures** -- no predicted class, no true class, no
   confidence.

All four are addressed here. Panels are drawn for correct and incorrect cases
alike; for a misclassification both the predicted-class and true-class maps are
shown side by side.

Usage:
    ./run.sh python gradcam_report.py --model ../compression/artifacts/densenet121_baseline
"""
import argparse
import json
import os

import numpy as np

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
SEED = 42


def flatten_model(model):
    """Expose the backbone's convolutions at the top level for Grad-CAM.

    A Sequential wrapping a functional backbone hides every conv layer inside one
    opaque entry, and tf-keras-vis then fails with "Graph disconnected".
    """
    import tensorflow as tf

    if not isinstance(model.layers[0], tf.keras.Model):
        return model
    backbone = model.layers[0]
    x = backbone.output
    for layer in model.layers[1:]:
        x = layer(x)
    return tf.keras.Model(inputs=backbone.input, outputs=x)


def preprocessing_for(model_dir):
    import tensorflow as tf

    if "mobilenet" in model_dir.lower():
        return tf.keras.applications.mobilenet_v2.preprocess_input
    return tf.keras.applications.densenet.preprocess_input


def overlay(axis, raw, cam, title, colour):
    from matplotlib import cm

    axis.imshow(raw)
    if cam is not None:
        axis.imshow(np.uint8(cm.jet(cam)[..., :3] * 255), alpha=0.45)
    axis.set_title(title, fontsize=8, color=colour)
    axis.set_xticks([])
    axis.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="../compression/artifacts/densenet121_baseline")
    parser.add_argument("--manifest", default="../compression/data/prepared/manifest_test.csv")
    parser.add_argument("--images", default="../compression/data/prepared/images")
    parser.add_argument("--n-correct", type=int, default=2)
    parser.add_argument("--n-incorrect", type=int, default=2)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore

    if not os.path.isdir(args.model):
        raise SystemExit("model not found: %s (train it first)" % args.model)

    model = flatten_model(tf.keras.models.load_model(args.model))
    preprocess = preprocessing_for(args.model)
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)

    frame = pd.read_csv(args.manifest)

    # Score the whole test split once so correct and incorrect cases can be
    # chosen deliberately rather than by whatever sorts first.
    print("Scoring %d test images ..." % len(frame))
    raws, probs = [], []
    for path in frame["file"]:
        raw = img_to_array(load_img(os.path.join(args.images, path),
                                    target_size=(224, 224))) / 255.0
        raws.append(raw)
        probs.append(model.predict(preprocess(raw[np.newaxis, ...] * 255.0), verbose=0)[0])
    probs = np.stack(probs)
    predicted = probs.argmax(axis=1)
    truth = frame["label"].to_numpy()
    print("  accuracy %.4f" % (predicted == truth).mean())

    rng = np.random.RandomState(SEED)
    records = []

    for cls_index, cls in enumerate(CLASSES):
        in_class = np.where(truth == cls_index)[0]
        hits = in_class[predicted[in_class] == cls_index]
        misses = in_class[predicted[in_class] != cls_index]

        chosen_hits = rng.choice(hits, min(args.n_correct, len(hits)), replace=False)
        chosen_misses = rng.choice(misses, min(args.n_incorrect, len(misses)), replace=False)
        chosen = list(chosen_hits) + list(chosen_misses)
        if not chosen:
            continue

        # One row per image. Column 1 is the image, column 2 the map for the
        # predicted class; for a mistake, column 3 adds the map for the true
        # class, which is what makes the error interpretable.
        columns = 3 if len(chosen_misses) else 2
        figure, axes = plt.subplots(len(chosen), columns,
                                    figsize=(3.2 * columns, 3.0 * len(chosen)))
        axes = np.atleast_2d(axes)

        for row, index in enumerate(chosen):
            raw = raws[index]
            batch = preprocess(raw[np.newaxis, ...] * 255.0)
            pred = int(predicted[index])
            correct = pred == cls_index
            confidence = float(probs[index][pred])

            cam_pred = np.asarray(gradcam(CategoricalScore([pred]), batch,
                                          penultimate_layer=-1)[0], dtype=np.float64)
            colour = "darkgreen" if correct else "firebrick"
            overlay(axes[row][0], raw, None,
                    "%s\ntrue: %s" % (os.path.basename(frame["file"][index]), cls), "black")
            overlay(axes[row][1], raw, cam_pred,
                    "predicted: %s\nconfidence %.3f%s"
                    % (CLASSES[pred], confidence, "" if correct else "  (wrong)"), colour)

            cam_true = None
            if columns == 3:
                if correct:
                    axes[row][2].axis("off")
                else:
                    cam_true = np.asarray(gradcam(CategoricalScore([cls_index]), batch,
                                                  penultimate_layer=-1)[0], dtype=np.float64)
                    overlay(axes[row][2], raw, cam_true,
                            "evidence for true class\n%s (p=%.3f)"
                            % (cls, float(probs[index][cls_index])), "navy")

            records.append({
                "file": frame["file"][index], "true_class": cls,
                "predicted": CLASSES[pred], "correct": bool(correct),
                "confidence": confidence,
                "probabilities": {c: float(probs[index][i]) for i, c in enumerate(CLASSES)},
            })

        figure.suptitle("Grad-CAM++ -- %s  (%d correct, %d incorrect)"
                        % (cls, len(chosen_hits), len(chosen_misses)), fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "gradcam_%s.png" % cls)
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print("  wrote %s  (%d correct, %d incorrect)"
              % (path, len(chosen_hits), len(chosen_misses)))

    summary = {
        "model": args.model,
        "test_accuracy": float((predicted == truth).mean()),
        "panels": records,
    }
    with open(os.path.join(args.out, "gradcam_report.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsaved -> %s" % os.path.join(args.out, "gradcam_report.json"))


if __name__ == "__main__":
    main()
