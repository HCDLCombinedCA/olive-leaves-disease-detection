"""Train a CNN from scratch, on the same data and protocol as the baselines.

RQ1 compares a from-scratch CNN, transfer learning, and glass-box models. The
transfer-learning and glass-box halves are covered by `train_baseline.py` and
`analysis/glassbox_fixed.py`; this fills the remaining gap. The scratch result
that exists in `Olive_Leaf_CNN_Model_Notebook.ipynb` cannot be used for that
comparison, because it was produced on the uncorrected data with a random
`validation_split=0.3` that scatters duplicate and burst photographs across the
train/validation boundary.

Two architectures are provided, and the difference between them is itself a
reportable result:

* **original** -- a faithful reproduction of the notebook's `modelBest`
  (cell 16): two 5x5 convolutions, spatial dropout, then `Flatten` into
  `Dense(64)`. Reproduced exactly so that the only thing that changes is the
  dataset and the evaluation protocol. Its 53x53x32 flatten costs 5,752,896
  parameters, 99.75% of the model, for two convolutions' worth of features.
* **improved** -- what the same budget buys when spent on depth instead: a
  strided stem, four convolutional stages with BatchNorm, and global average
  pooling in place of `Flatten`. Roughly a tenth of the parameters.

Everything else is deliberately shared with the transfer-learning baselines via
`common`: the leakage-free manifests, 224x224 inputs, training-only
augmentation, balanced class weights, and the accuracy/macro-F1/per-class-recall
evaluation. That is what makes the numbers comparable; the architecture is the
only variable.

Usage:
    ./run.sh python src/train_scratch.py --arch original
    ./run.sh python src/train_scratch.py --arch improved
"""
import argparse
import os
import time

import tensorflow as tf

import common

# The notebook rescaled to [0, 1] with `ImageDataGenerator(rescale=1./255)`.
# Kept identical here: a scratch network has no pretrained statistics to match,
# so this is a free choice, and matching the notebook keeps the reproduction
# honest.
def preprocess_input(batch):
    return batch / 255.0


def build_original(dropout=0.2):
    """`modelBest` from Olive_Leaf_CNN_Model_Notebook.ipynb, cell 16.

    Layer-for-layer identical, including the 5x5 kernels, the valid padding that
    Keras applies by default, and the spatial dropout rate. With a 224x224 input
    the second pooling layer emits 53x53x32, which `Flatten` turns into an 89,888
    dimensional vector -- hence the 5.75M-parameter dense layer.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (5, 5), activation="relu",
                               input_shape=common.IMAGE_SIZE + (3,)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(dropout),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(common.NUM_CLASSES, activation="softmax"),
    ], name="scratch_original")


def build_improved(dropout=0.4):
    """A conventionally structured scratch CNN of the same input resolution.

    Three changes carry the difference: BatchNorm so a network this deep trains
    at all without pretrained weights, a stride-2 stem so the expensive early
    layers run at 112x112 rather than 224x224, and global average pooling
    instead of `Flatten` -- which is what removes the 5.75M-parameter head.
    """
    def block(filters, count):
        layers = []
        for _ in range(count):
            layers += [
                tf.keras.layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        return layers

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False,
                               input_shape=common.IMAGE_SIZE + (3,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        *block(32, 1),    # 112 -> 56
        *block(64, 2),    # 56 -> 28
        *block(128, 2),   # 28 -> 14
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(common.NUM_CLASSES, activation="softmax"),
    ], name="scratch_improved")


ARCHITECTURES = {"original": build_original, "improved": build_improved}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="improved")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="reproduce the notebook, which trained unweighted on "
                             "an imbalanced set; off by default so the scratch run "
                             "matches the transfer-learning baselines")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    if args.threads:
        common.limit_threads(args.threads)
    common.set_seeds()

    model = ARCHITECTURES[args.arch]()
    train_gen, val_gen, test_gen = common.make_generators(preprocess_input, args.batch_size)
    weights = None if args.no_class_weights else common.class_weights(train_gen)
    model.summary()
    if weights:
        print("class weights: %s" % {common.CLASSES[k]: round(v, 3) for k, v in weights.items()})

    # Single stage: there are no pretrained features to protect, so the staged
    # freeze/unfreeze schedule the transfer-learning baselines use does not apply.
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print("\n=== training %s from scratch (%d epochs max) ===" % (args.arch, args.epochs))
    started = time.time()
    history = model.fit(
        train_gen, epochs=args.epochs, validation_data=val_gen,
        class_weight=weights, workers=args.workers, use_multiprocessing=False,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        ])
    elapsed = time.time() - started

    predict = lambda batch: model.predict(batch, verbose=0)
    results = {}
    for name, generator in (("val", val_gen), ("test", test_gen)):
        y_true, y_pred, _ = common.predict_generator(predict, generator)
        results[name] = common.evaluate(y_true, y_pred)
        print(common.format_metrics("scratch_%s (%s)" % (args.arch, name), results[name]))

    model_dir = os.path.join(args.out, "scratch_%s_baseline" % args.arch)
    model.save(model_dir, include_optimizer=False)
    print("\nsaved model -> %s" % model_dir)

    common.save_json({
        "backbone": "scratch_%s" % args.arch,
        "trained_from": "random initialisation",
        "seed": common.SEED,
        "batch_size": args.batch_size,
        "class_weights": "balanced" if weights else "none",
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(elapsed, 1),
        "total_parameters": int(model.count_params()),
        "metrics": results,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }, os.path.join("results", "scratch_%s_baseline.json" % args.arch))
    print("saved metrics -> results/scratch_%s_baseline.json" % args.arch)


if __name__ == "__main__":
    main()
