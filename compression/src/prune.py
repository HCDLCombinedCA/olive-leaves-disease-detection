"""Magnitude pruning implemented directly with Keras weight access.

`tensorflow_model_optimization` is not installed in the course image, so pruning
here is written with `get_weights()` / `set_weights()` and a training callback --
primitives from Weeks 1-4. That keeps the work inside the course toolset and has
the side benefit of making the mechanism explicit: the report can state exactly
which tensors were masked and what fraction of each was zeroed, rather than
pointing at a wrapper.

How it works
------------
1. For every prunable layer, take the absolute values of its kernel and find the
   quantile corresponding to the target sparsity. Weights below it are set to
   zero and recorded in a binary mask.
2. Fine-tune. After **every training batch** the masks are re-applied, because
   the optimiser would otherwise immediately push pruned weights away from zero
   again. This callback is the manual equivalent of TFMOT's `UpdatePruningStep`,
   and omitting it is the classic way to end up with a model that reports a
   sparsity target but is not actually sparse.

What gets pruned
----------------
Conv2D and Dense kernels only.

* **Biases are left alone** -- they are a negligible share of the parameters and
  zeroing them shifts every activation.
* **DepthwiseConv2D is excluded.** In MobileNetV2 those layers hold very few
  weights each but sit on every information path, so pruning them costs far more
  accuracy per parameter saved than pruning the pointwise convolutions.
* **BatchNorm is excluded** -- its parameters are scale/shift statistics, not
  connections.

A note on file size
-------------------
Unstructured pruning zeroes weights but does not change tensor shapes, so the
SavedModel and .tflite files do **not** shrink. The benefit shows up only after
entropy coding, which is why `measure.py` reports gzipped size alongside raw
size. This is the honest framing, and it is exactly the trade-off Kuzmin et al.
(2023) analyse when arguing quantisation is usually the better return.

Usage:
    ./run.sh python src/prune.py --backbone mobilenetv2 --sparsity 0.5
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import common

PRUNABLE = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)
EXCLUDED = (tf.keras.layers.DepthwiseConv2D,)


def iter_prunable_layers(model):
    """Walk nested models, yielding layers whose kernels should be pruned."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for inner in iter_prunable_layers(layer):
                yield inner
        elif isinstance(layer, EXCLUDED):
            continue
        elif isinstance(layer, PRUNABLE) and getattr(layer, "kernel", None) is not None:
            yield layer


def compute_masks(model, sparsity):
    """Per-layer magnitude masks.

    Sparsity is applied per layer rather than globally so that no single layer is
    wiped out: a global threshold tends to remove almost everything from layers
    whose weights are small in absolute terms, even when those weights matter.
    """
    masks = {}
    for layer in iter_prunable_layers(model):
        kernel = layer.kernel.numpy()
        threshold = np.quantile(np.abs(kernel), sparsity)
        masks[layer.name] = (np.abs(kernel) > threshold).astype(kernel.dtype)
    return masks


def apply_masks(model, masks):
    for layer in iter_prunable_layers(model):
        mask = masks.get(layer.name)
        if mask is not None:
            layer.kernel.assign(layer.kernel * mask)


class ReapplyMasks(tf.keras.callbacks.Callback):
    """Re-zero pruned weights after every batch, keeping sparsity fixed."""

    def __init__(self, masks, layers):
        super(ReapplyMasks, self).__init__()
        self.masks = masks
        self.layers = layers

    def on_train_batch_end(self, batch, logs=None):
        for layer in self.layers:
            mask = self.masks.get(layer.name)
            if mask is not None:
                layer.kernel.assign(layer.kernel * mask)


def sparsity_report(model):
    """Measured sparsity, so the report quotes what happened rather than the target."""
    total = zeros = 0
    per_layer = {}
    for layer in iter_prunable_layers(model):
        kernel = layer.kernel.numpy()
        layer_zeros = int((kernel == 0).sum())
        per_layer[layer.name] = {
            "parameters": int(kernel.size),
            "zeros": layer_zeros,
            "sparsity": round(layer_zeros / kernel.size, 4),
        }
        total += kernel.size
        zeros += layer_zeros
    return {
        "prunable_parameters": int(total),
        "prunable_zeros": int(zeros),
        "overall_sparsity": round(zeros / total, 4) if total else 0.0,
        "per_layer": per_layer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="target fraction of kernel weights to zero, per layer")
    parser.add_argument("--epochs", type=int, default=12, help="fine-tuning epochs after pruning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    common.set_seeds()
    _constructor, preprocess_fn, _prefix = common.BACKBONES[args.backbone]
    train_gen, val_gen, test_gen = common.make_generators(preprocess_fn, args.batch_size)

    baseline_dir = os.path.join(args.out, "%s_baseline" % args.backbone)
    if not os.path.isdir(baseline_dir):
        raise SystemExit("baseline not found: %s (train it first)" % baseline_dir)
    model = tf.keras.models.load_model(baseline_dir)

    predict = lambda batch: model.predict(batch, verbose=0)
    stages = {}

    y_true, y_pred, _ = common.predict_generator(predict, val_gen)
    stages["baseline"] = common.evaluate(y_true, y_pred)
    print(common.format_metrics("baseline (val)", stages["baseline"]))

    # ---- Prune ----
    print("\n=== pruning to %.0f%% per-layer sparsity ===" % (args.sparsity * 100))
    masks = compute_masks(model, args.sparsity)
    apply_masks(model, masks)
    report_before = sparsity_report(model)
    print("  %d prunable weights across %d layers, measured sparsity %.1f%%"
          % (report_before["prunable_parameters"], len(masks),
             report_before["overall_sparsity"] * 100))

    y_true, y_pred, _ = common.predict_generator(predict, val_gen)
    stages["pruned_no_finetune"] = common.evaluate(y_true, y_pred)
    print(common.format_metrics("pruned, no fine-tune (val)", stages["pruned_no_finetune"]))

    # ---- Fine-tune with masks held ----
    print("\n=== fine-tuning %d epochs, masks re-applied after every batch ===" % args.epochs)
    layers = list(iter_prunable_layers(model))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen,
              class_weight=common.class_weights(train_gen),
              callbacks=[
                  ReapplyMasks(masks, layers),
                  tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
              ],
              workers=args.workers, use_multiprocessing=False, verbose=2)

    # EarlyStopping restores the best weights, which may predate the last mask
    # application, so re-apply before measuring and saving.
    apply_masks(model, masks)
    report_after = sparsity_report(model)
    print("  sparsity after fine-tuning: %.1f%%" % (report_after["overall_sparsity"] * 100))

    for split_name, generator in (("val", val_gen), ("test", test_gen)):
        y_true, y_pred, _ = common.predict_generator(predict, generator)
        stages["pruned_finetuned_%s" % split_name] = common.evaluate(y_true, y_pred)
        print(common.format_metrics("pruned + fine-tuned (%s)" % split_name,
                                    stages["pruned_finetuned_%s" % split_name]))

    tag = "pruned%02d" % round(args.sparsity * 100)
    model_dir = os.path.join(args.out, "%s_%s" % (args.backbone, tag))
    model.save(model_dir, include_optimizer=False)
    print("\nsaved model -> %s" % model_dir)

    common.save_json({
        "backbone": args.backbone,
        "target_sparsity": args.sparsity,
        "epochs": args.epochs,
        "sparsity_before_finetune": report_before,
        "sparsity_after_finetune": report_after,
        "metrics": stages,
    }, os.path.join("results", "%s_%s.json" % (args.backbone, tag)))
    print("saved metrics -> results/%s_%s.json" % (args.backbone, tag))


if __name__ == "__main__":
    main()
