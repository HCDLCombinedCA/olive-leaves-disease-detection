"""Measure every model variant on one footing and emit the results table.

Reported per variant:

* **parameters** -- total, and how many are non-zero (the two diverge once a
  model has been pruned).
* **file size** -- the artefact as it would ship.
* **gzipped size** -- the same file after entropy coding. Unstructured pruning
  does not change tensor shapes, so it shows no benefit in raw size; the saving
  appears only here. Reporting both is what makes the pruning result honest
  rather than disappointing.
* **CPU latency** -- median seconds for a single image, single-threaded.
* **throughput** -- images per second derived from the median.
* **accuracy / macro-F1 / per-class recall** on validation and test.

Latency is measured single-threaded with a fixed thread pool and a warm-up
phase, and reported as a median rather than a mean: the first invocation of any
runtime carries graph-construction cost, which drags the mean around. The same
convention is used by `leaf_segmenter/eval/evaluate_timings.py` in this repo.

Usage:
    ./run.sh python src/measure.py --backbone mobilenetv2
"""
import argparse
import csv
import glob
import gzip
import os
import shutil
import tempfile
import time

import numpy as np
import tensorflow as tf

import common
from quantise import tflite_predictor

WARMUP_RUNS = 10
TIMED_RUNS = 100


def directory_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def gzipped_size(path):
    """Size after gzip. For a SavedModel directory, the variables file dominates,
    so the whole tree is compressed as a tar-free concatenation of its files."""
    total = 0
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = [os.path.join(r, f) for r, _d, fs in os.walk(path) for f in fs]
    for item in paths:
        with open(item, "rb") as src:
            raw = src.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as tmp:
            tmp_path = tmp.name
        with gzip.open(tmp_path, "wb", compresslevel=9) as dst:
            dst.write(raw)
        total += os.path.getsize(tmp_path)
        os.unlink(tmp_path)
    return total


def keras_parameter_stats(model_dir):
    model = tf.keras.models.load_model(model_dir)
    total = int(model.count_params())
    nonzero = 0
    for weight in model.weights:
        nonzero += int(np.count_nonzero(weight.numpy()))
    return total, nonzero, model


def measure_latency(predict, sample):
    """Median single-image latency in milliseconds."""
    for _ in range(WARMUP_RUNS):
        predict(sample)
    timings = []
    for _ in range(TIMED_RUNS):
        started = time.perf_counter()
        predict(sample)
        timings.append(time.perf_counter() - started)
    timings = np.array(timings)
    return {
        "median_ms": float(np.median(timings) * 1000),
        "mean_ms": float(timings.mean() * 1000),
        "p90_ms": float(np.percentile(timings, 90) * 1000),
        "throughput_img_s": float(1.0 / np.median(timings)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1,
                        help="thread pool size for latency measurement")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    common.limit_threads(args.threads)
    common.set_seeds()
    _constructor, preprocess_fn, _prefix = common.BACKBONES[args.backbone]
    _train_gen, val_gen, test_gen = common.make_generators(
        preprocess_fn, args.batch_size, augment=False)

    # One preprocessed image, reused for every latency measurement so the
    # variants are timed on identical input.
    val_gen.reset()
    sample = val_gen[0][0][:1]

    variants = []

    # Keras SavedModel directories (baseline and any pruned versions)
    for model_dir in sorted(glob.glob(os.path.join(args.artifacts, "%s_*" % args.backbone))):
        if not os.path.isdir(model_dir):
            continue
        variants.append(("keras", os.path.basename(model_dir), model_dir))

    # tf.lite artefacts
    for path in sorted(glob.glob(os.path.join(args.artifacts, "%s_*.tflite" % args.backbone))):
        variants.append(("tflite", os.path.basename(path)[:-len(".tflite")], path))

    if not variants:
        raise SystemExit("no artefacts found under %s for %s" % (args.artifacts, args.backbone))

    rows = []
    for kind, name, path in variants:
        print("\n=== %s (%s) ===" % (name, kind))
        if kind == "keras":
            total, nonzero, model = keras_parameter_stats(path)
            predict = lambda batch: model.predict(batch, verbose=0)
        else:
            predict = tflite_predictor(path, num_threads=args.threads)
            # Parameter counts come from the source Keras model; a .tflite file
            # stores packed tensors and does not expose a comparable count.
            total = nonzero = None

        latency = measure_latency(predict, sample)
        print("  latency median %.1f ms  (mean %.1f, p90 %.1f)  -> %.1f img/s"
              % (latency["median_ms"], latency["mean_ms"], latency["p90_ms"],
                 latency["throughput_img_s"]))

        metrics = {}
        for split_name, generator in (("val", val_gen), ("test", test_gen)):
            y_true, y_pred, _ = common.predict_generator(predict, generator)
            metrics[split_name] = common.evaluate(y_true, y_pred)
            print("  " + common.format_metrics(split_name, metrics[split_name]))

        raw = directory_size(path)
        gz = gzipped_size(path)
        print("  size %.2f MB  ->  gzip %.2f MB" % (raw / 1e6, gz / 1e6))

        rows.append({
            "variant": name,
            "kind": kind,
            "parameters": total,
            "nonzero_parameters": nonzero,
            "sparsity": (round(1 - nonzero / total, 4) if total else None),
            "size_mb": round(raw / 1e6, 3),
            "gzip_mb": round(gz / 1e6, 3),
            "latency_median_ms": round(latency["median_ms"], 2),
            "throughput_img_s": round(latency["throughput_img_s"], 1),
            "val_macro_f1": round(metrics["val"]["macro_f1"], 4),
            "test_accuracy": round(metrics["test"]["accuracy"], 4),
            "test_macro_f1": round(metrics["test"]["macro_f1"], 4),
            "test_recall_healthy": round(metrics["test"]["per_class_recall"]["Healthy"], 4),
            "test_recall_aculus": round(
                metrics["test"]["per_class_recall"]["aculus_olearius"], 4),
            "test_recall_peacock": round(
                metrics["test"]["per_class_recall"]["olive_peacock_spot"], 4),
        })

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "%s_compression_table.csv" % args.backbone)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n%-34s %9s %9s %9s %9s %9s" % (
        "variant", "size MB", "gzip MB", "ms/img", "test F1", "aculus R"))
    print("-" * 84)
    for row in rows:
        print("%-34s %9.2f %9.2f %9.1f %9.4f %9.4f" % (
            row["variant"], row["size_mb"], row["gzip_mb"],
            row["latency_median_ms"], row["test_macro_f1"], row["test_recall_aculus"]))
    print("\nsaved -> %s" % csv_path)


if __name__ == "__main__":
    main()
