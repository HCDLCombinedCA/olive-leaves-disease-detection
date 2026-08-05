"""Train the baseline classifier that the compression experiments start from.

Two backbones are supported:

* **mobilenetv2** (primary) -- the architecture the project proposal actually
  named, designed for edge inference, and roughly 10x cheaper than DenseNet121
  to train on CPU. Compressing a network that is already built for constrained
  hardware is the more interesting result for RQ2.
* **densenet121** (comparison) -- matches the existing notebook so the compressed
  numbers can be read against the result the team already has.

The training recipe follows the staged transfer-learning approach from the
existing notebook, which was already sound: freeze the backbone and train the
head, then unfreeze the last block and fine-tune at a much lower learning rate,
keeping BatchNorm layers in inference mode throughout.

Two things are fixed relative to the original:

* `set_seeds()` seeds weight initialisation, not just data shuffling.
* The model is saved with `include_optimizer=False`. The existing checkpoint in
  the repo is 69 MB, two thirds of which is Adam moment state -- that would show
  up as a fake 3x compression ratio in the results table.

Usage:
    ./run.sh python src/train_baseline.py --backbone mobilenetv2
"""
import argparse
import os
import time

import tensorflow as tf

import common


def build_model(backbone_name, dropout=0.3):
    """Backbone with a global-average-pooled classification head.

    Global average pooling rather than Flatten is deliberate. The scratch CNN in
    the existing notebook flattens a 53x53x32 feature map straight into Dense(64),
    which costs 5,752,896 parameters -- 99.75% of that model -- for two
    convolutional layers' worth of features. Pooling keeps the head at a few
    hundred parameters and is the standard choice for transfer learning.
    """
    constructor, preprocess_fn, fine_tune_prefix = common.BACKBONES[backbone_name]
    base = constructor(weights="imagenet", include_top=False,
                       input_shape=common.IMAGE_SIZE + (3,), pooling="avg")
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(common.NUM_CLASSES, activation="softmax"),
    ])
    return model, base, preprocess_fn, fine_tune_prefix


def unfreeze_top(base, prefix):
    """Unfreeze the final block only, keeping BatchNorm in inference mode.

    Letting BatchNorm update its statistics while most of the network is frozen
    destroys the pretrained features -- this is the single most common way staged
    fine-tuning goes wrong.
    """
    base.trainable = True
    unfrozen = 0
    for layer in base.layers:
        if not layer.name.startswith(prefix):
            layer.trainable = False
        else:
            unfrozen += 1
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return unfrozen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=35)
    parser.add_argument("--threads", type=int, default=0,
                        help="TensorFlow thread pool size; 0 lets TF decide")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel workers feeding the image generator")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    if args.threads:
        common.limit_threads(args.threads)
    common.set_seeds()

    model, base, preprocess_fn, fine_tune_prefix = build_model(args.backbone)
    train_gen, val_gen, test_gen = common.make_generators(preprocess_fn, args.batch_size)
    weights = common.class_weights(train_gen)
    print("class weights: %s" % {common.CLASSES[k]: round(v, 3) for k, v in weights.items()})
    print("trainable parameters (head only): %d" % sum(
        int(tf.size(w)) for w in model.trainable_weights))

    fit_common = dict(validation_data=val_gen, class_weight=weights,
                      workers=args.workers, use_multiprocessing=False, verbose=2)

    # ---- Stage 1: frozen backbone, train the head ----
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print("\n=== stage 1: frozen backbone (%d epochs max) ===" % args.head_epochs)
    started = time.time()
    history_head = model.fit(
        train_gen, epochs=args.head_epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)],
        **fit_common)

    # ---- Stage 2: unfreeze the last block, fine-tune slowly ----
    unfrozen = unfreeze_top(base, fine_tune_prefix)
    print("\n=== stage 2: fine-tuning %d layers matching '%s' (%d epochs max) ==="
          % (unfrozen, fine_tune_prefix, args.finetune_epochs))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    history_finetune = model.fit(
        train_gen, epochs=args.finetune_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1),
        ],
        **fit_common)
    elapsed = time.time() - started

    # ---- Evaluate ----
    predict = lambda batch: model.predict(batch, verbose=0)
    results = {}
    for name, generator in (("val", val_gen), ("test", test_gen)):
        y_true, y_pred, _ = common.predict_generator(predict, generator)
        results[name] = common.evaluate(y_true, y_pred)
        print(common.format_metrics("%s (%s)" % (args.backbone, name), results[name]))

    # ---- Save ----
    # SavedModel format, matching the Week 7 Production lab. include_optimizer is
    # off so the file measures the model, not the optimiser state.
    model_dir = os.path.join(args.out, "%s_baseline" % args.backbone)
    model.save(model_dir, include_optimizer=False)
    print("\nsaved model -> %s" % model_dir)

    common.save_json({
        "backbone": args.backbone,
        "seed": common.SEED,
        "batch_size": args.batch_size,
        "epochs_run": {
            "head": len(history_head.history["loss"]),
            "finetune": len(history_finetune.history["loss"]),
        },
        "train_seconds": round(elapsed, 1),
        "total_parameters": int(model.count_params()),
        "metrics": results,
        "history": {
            "head": {k: [float(x) for x in v] for k, v in history_head.history.items()},
            "finetune": {k: [float(x) for x in v] for k, v in history_finetune.history.items()},
        },
    }, os.path.join("results", "%s_baseline.json" % args.backbone))
    print("saved metrics -> results/%s_baseline.json" % args.backbone)


if __name__ == "__main__":
    main()
