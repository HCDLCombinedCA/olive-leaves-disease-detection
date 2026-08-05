"""Shared dataset, preprocessing and evaluation helpers.

Everything downstream (training, quantisation, pruning, measurement, LIME) goes
through this module so that a single definition of "how an image becomes model
input" is used everywhere. Getting that wrong is the most common cause of a
quantised model appearing to collapse, so it is defined once and reused.
"""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, recall_score

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = (224, 224)
SEED = 42

DATA_ROOT = "data/prepared"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
TRAINVAL_MANIFEST = os.path.join(DATA_ROOT, "manifest_trainval.csv")
TEST_MANIFEST = os.path.join(DATA_ROOT, "manifest_test.csv")

# Backbone -> (keras application constructor, matching preprocess_input).
# The preprocessing function is part of the backbone's identity: MobileNetV2
# expects inputs scaled to [-1, 1] while DenseNet uses torch-style channel
# normalisation. Pairing them incorrectly silently destroys accuracy.
BACKBONES = {
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
        # Last inverted-residual stage; unfrozen during fine-tuning.
        "block_16",
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
        "conv5",
    ),
}


def set_seeds(seed=SEED):
    """Seed Python, numpy and TensorFlow.

    The original notebook only passed `seed=` to the data generators, which fixes
    shuffling but leaves weight initialisation random. Sixteen architecture
    experiments were run that way, and the final model landed on an unlucky
    initialisation that collapsed onto two of the three classes.
    """
    tf.keras.utils.set_random_seed(seed)


def limit_threads(num_threads):
    """Pin TensorFlow's thread pools.

    Required for latency numbers to be reproducible; also called during training
    so runs are comparable.
    """
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)


def load_manifest(path):
    import pandas as pd
    return pd.read_csv(path)


def make_generators(preprocess_fn, batch_size=32, augment=True):
    """Build train / val / test iterators from the prepared manifests.

    Augmentation is applied to the training split only. Validation and test are
    preprocessed but never augmented, so their scores reflect the real input
    distribution.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    trainval = load_manifest(TRAINVAL_MANIFEST)
    test = load_manifest(TEST_MANIFEST)

    train_df = trainval[trainval["split"] == "train"]
    val_df = trainval[trainval["split"] == "val"]

    train_args = dict(preprocessing_function=preprocess_fn)
    if augment:
        train_args.update(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
    train_gen = ImageDataGenerator(**train_args).flow_from_dataframe(
        train_df, directory=IMAGES_DIR, x_col="file", y_col="class",
        classes=CLASSES, class_mode="categorical", target_size=IMAGE_SIZE,
        batch_size=batch_size, shuffle=True, seed=SEED)

    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    val_gen = eval_gen.flow_from_dataframe(
        val_df, directory=IMAGES_DIR, x_col="file", y_col="class",
        classes=CLASSES, class_mode="categorical", target_size=IMAGE_SIZE,
        batch_size=batch_size, shuffle=False)
    test_gen = eval_gen.flow_from_dataframe(
        test, directory=IMAGES_DIR, x_col="file", y_col="class",
        classes=CLASSES, class_mode="categorical", target_size=IMAGE_SIZE,
        batch_size=batch_size, shuffle=False)
    return train_gen, val_gen, test_gen


def class_weights(train_gen):
    """Balanced class weights for the training split.

    The dataset is imbalanced (763 / 578 / 547 after de-duplication) and the
    minority class is the one the original scratch CNN failed to learn at all.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def predict_generator(predict_fn, generator):
    """Run a prediction callable over a generator, returning (y_true, y_pred, probs).

    `predict_fn` takes a batch of preprocessed images and returns class
    probabilities. Accepting a callable rather than a Keras model lets the same
    evaluation path serve Keras models and tf.lite interpreters alike, which is
    what makes the compressed variants directly comparable to the baseline.
    """
    generator.reset()
    probs = []
    steps = len(generator)
    for i in range(steps):
        batch_x, _ = generator[i]
        probs.append(predict_fn(batch_x))
    probs = np.concatenate(probs, axis=0)
    y_true = generator.classes[:len(probs)]
    return y_true, probs.argmax(axis=1), probs


def evaluate(y_true, y_pred):
    """Metrics reported for every model variant.

    Per-class recall matters more than accuracy here: quantisation tends to hurt
    the minority class disproportionately, and overall accuracy hides that.
    """
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_recall": {
            cls: float(score) for cls, score in zip(
                CLASSES, recall_score(y_true, y_pred, average=None,
                                      labels=list(range(NUM_CLASSES))))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def format_metrics(name, metrics):
    recalls = "  ".join("%s=%.3f" % (c[:8], metrics["per_class_recall"][c]) for c in CLASSES)
    return "%-28s acc=%.4f  macro-F1=%.4f  |  %s" % (
        name, metrics["accuracy"], metrics["macro_f1"], recalls)
