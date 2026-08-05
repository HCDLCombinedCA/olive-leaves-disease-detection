"""Sparsity sweep: how much can be pruned before accuracy falls away?

The single 50% run in `prune.py` left an open question. Its recovery curve was
still climbing at the final epoch (val accuracy 0.598 -> 0.755 over 12 epochs,
no plateau), so the resulting macro-F1 measures "12 epochs of recovery", not
"the cost of 50% sparsity". This sweep answers the question properly by varying
sparsity and letting each point train to convergence.

Two changes to the recovery protocol, both of which matter:

* **Learning rate 1e-4 rather than 1e-5.** The baseline's second stage used 1e-5
  because it was gently adapting pretrained features. Pruning half the weights is
  major surgery, and recovering from it needs a rate an order of magnitude
  higher. The slow, still-climbing curve in the original run is the signature of
  a rate that is too low for the job.
* **30 epochs with early stopping and LR reduction**, so each point stops when it
  has actually converged instead of when the epoch budget runs out.

Running the same sweep on both backbones tests a specific hypothesis: MobileNetV2
is built from depthwise separable convolutions and is already parameter-efficient,
so it should have less redundancy to give up than DenseNet121, which carries
three times the parameters for less than a point of extra accuracy. If that holds,
the compact architecture should degrade faster as sparsity rises.

Usage:
    ./run.sh python src/prune_sweep.py --backbone mobilenetv2
    ./run.sh python src/prune_sweep.py --backbone densenet121 --sparsities 0.2 0.35 0.5
"""
import argparse
import csv
import os

import tensorflow as tf

import common
from prune import (ReapplyMasks, apply_masks, compute_masks, iter_prunable_layers,
                   sparsity_report)


def run_one(backbone, sparsity, train_gen, val_gen, test_gen, weights, args):
    """Prune a fresh copy of the baseline to one sparsity level and recover it."""
    baseline_dir = os.path.join(args.out, "%s_baseline" % backbone)
    model = tf.keras.models.load_model(baseline_dir)
    predict = lambda batch: model.predict(batch, verbose=0)

    print("\n=== %s @ %.0f%% sparsity ===" % (backbone, sparsity * 100))
    masks = compute_masks(model, sparsity)
    apply_masks(model, masks)
    report = sparsity_report(model)
    print("  measured sparsity %.1f%% over %d prunable weights"
          % (report["overall_sparsity"] * 100, report["prunable_parameters"]))

    y_true, y_pred, _ = common.predict_generator(predict, val_gen)
    before = common.evaluate(y_true, y_pred)
    print("  " + common.format_metrics("pruned, before recovery (val)", before))

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_gen, epochs=args.epochs, validation_data=val_gen, class_weight=weights,
        callbacks=[
            ReapplyMasks(masks, list(iter_prunable_layers(model))),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=3, min_lr=1e-7, verbose=0),
        ],
        workers=args.workers, use_multiprocessing=False, verbose=2)

    # EarlyStopping restores the best weights, which may predate the last mask
    # application, so re-apply before measuring and saving.
    apply_masks(model, masks)
    after = sparsity_report(model)

    metrics = {}
    for split_name, generator in (("val", val_gen), ("test", test_gen)):
        y_true, y_pred, _ = common.predict_generator(predict, generator)
        metrics[split_name] = common.evaluate(y_true, y_pred)
        print("  " + common.format_metrics("recovered (%s)" % split_name, metrics[split_name]))

    tag = "sweep%02d" % round(sparsity * 100)
    model_dir = os.path.join(args.out, "%s_%s" % (backbone, tag))
    model.save(model_dir, include_optimizer=False)

    return {
        "target_sparsity": sparsity,
        "measured_sparsity": after["overall_sparsity"],
        "epochs_run": len(history.history["loss"]),
        "val_macro_f1_before_recovery": before["macro_f1"],
        "val_macro_f1": metrics["val"]["macro_f1"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_recall_healthy": metrics["test"]["per_class_recall"]["Healthy"],
        "test_recall_aculus": metrics["test"]["per_class_recall"]["aculus_olearius"],
        "test_recall_peacock": metrics["test"]["per_class_recall"]["olive_peacock_spot"],
        "model_dir": model_dir,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--sparsities", nargs="+", type=float,
                        default=[0.2, 0.35, 0.5, 0.65])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    common.set_seeds()
    _constructor, preprocess_fn, _prefix = common.BACKBONES[args.backbone]
    train_gen, val_gen, test_gen = common.make_generators(preprocess_fn, args.batch_size)
    weights = common.class_weights(train_gen)

    # Baseline metrics are shown as the 0% row so the curve has an anchor.
    baseline_metrics = None
    path = os.path.join("results", "%s_baseline.json" % args.backbone)
    if os.path.isfile(path):
        import json
        with open(path) as fh:
            baseline_metrics = json.load(fh)["metrics"]

    rows = []
    for sparsity in args.sparsities:
        rows.append(run_one(args.backbone, sparsity, train_gen, val_gen, test_gen,
                            weights, args))

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "%s_pruning_sweep.csv" % args.backbone)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n%-10s %10s %10s %14s %12s %12s" % (
        "sparsity", "measured", "epochs", "before recov.", "val F1", "test F1"))
    print("-" * 74)
    if baseline_metrics:
        print("%-10s %10s %10s %14s %12.4f %12.4f" % (
            "0% (base)", "--", "--", "--",
            baseline_metrics["val"]["macro_f1"], baseline_metrics["test"]["macro_f1"]))
    for row in rows:
        print("%-10.0f%% %9.1f%% %10d %14.4f %12.4f %12.4f" % (
            row["target_sparsity"] * 100, row["measured_sparsity"] * 100,
            row["epochs_run"], row["val_macro_f1_before_recovery"],
            row["val_macro_f1"], row["test_macro_f1"]))
    print("\nsaved -> %s" % csv_path)


if __name__ == "__main__":
    main()
