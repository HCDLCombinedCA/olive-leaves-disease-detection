"""Post-training quantisation with tf.lite.

Three variants, all produced by the converter that ships with TensorFlow 2.12 --
core TensorFlow, no add-on package:

* **dynamic range** -- weights stored as int8, activations computed in float.
  One converter flag, no calibration data, no retraining. Roughly 4x smaller.
* **float16** -- weights stored as half precision. Roughly 2x smaller and
  essentially lossless; the natural choice for an edge target with a GPU or NPU
  that handles fp16 natively.
* **full integer** -- weights *and* activations int8, which is what a
  microcontroller or an integer-only accelerator needs. Requires a representative
  dataset so the converter can calibrate activation ranges.

The representative dataset draws from the **validation** split, never test:
calibration is a form of fitting, and using test data here would leak.

Every converted model is immediately evaluated on the validation and test splits
through the same `common.predict_generator` path as the Keras baseline, so the
numbers are directly comparable. Quantisation failures are usually silent -- the
model converts fine and simply predicts badly -- so verifying rather than
assuming is the point.

Usage:
    ./run.sh python src/quantise.py --backbone mobilenetv2
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import common

# How many batches of validation images to feed the converter for calibration.
# A few hundred images is ample for activation range estimation.
CALIBRATION_BATCHES = 10


def tflite_predictor(model_path, num_threads=1):
    """Wrap a .tflite file as a batch -> probabilities callable.

    The interpreter takes one image at a time, so batches are looped. Returning
    the same shape as `model.predict` lets the evaluation and LIME code treat
    Keras models and quantised models identically.

    Input/output quantisation parameters are applied here when present, so a
    fully-integer model still accepts float images and returns float
    probabilities to the caller.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]

    def predict(batch):
        outputs = []
        for image in batch:
            x = image[np.newaxis, ...]
            if in_scale:                       # fully-integer input
                x = np.round(x / in_scale + in_zero).astype(input_detail["dtype"])
            else:
                x = x.astype(input_detail["dtype"])
            interpreter.set_tensor(input_detail["index"], x)
            interpreter.invoke()
            y = interpreter.get_tensor(output_detail["index"])[0]
            if out_scale:                      # dequantise back to probabilities
                y = (y.astype(np.float32) - out_zero) * out_scale
            outputs.append(y)
        return np.stack(outputs)

    return predict


def representative_dataset(val_gen, batches=CALIBRATION_BATCHES):
    """Yield single preprocessed images for activation-range calibration.

    Uses the same preprocessing as training. A mismatch here is the usual reason
    a fully-integer model collapses.
    """
    def generator():
        val_gen.reset()
        for i in range(min(batches, len(val_gen))):
            batch_x, _ = val_gen[i]
            for image in batch_x:
                yield [image[np.newaxis, ...].astype(np.float32)]
    return generator


def convert(model_dir, variant, val_gen):
    """Produce the .tflite bytes for one quantisation variant."""
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

    if variant == "float32":
        pass                                    # no optimisation: the TFLite control
    elif variant == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "full_integer":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(val_gen)
        # Restricting to the int8 op set makes the converter fail loudly if any
        # layer cannot be quantised, rather than silently leaving it in float.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError("unknown variant %r" % variant)

    return converter.convert()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(common.BACKBONES), default="mobilenetv2")
    parser.add_argument("--model-dir", default=None,
                        help="SavedModel to quantise; defaults to the backbone's baseline")
    parser.add_argument("--tag", default="baseline",
                        help="label for the source model, used in output filenames")
    parser.add_argument("--variants", nargs="+",
                        default=["float32", "dynamic_range", "float16", "full_integer"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    common.set_seeds()
    _constructor, preprocess_fn, _prefix = common.BACKBONES[args.backbone]
    _train_gen, val_gen, test_gen = common.make_generators(
        preprocess_fn, args.batch_size, augment=False)

    model_dir = args.model_dir or os.path.join(args.out, "%s_%s" % (args.backbone, args.tag))
    if not os.path.isdir(model_dir):
        raise SystemExit("model not found: %s (train it first)" % model_dir)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for variant in args.variants:
        print("\n=== converting: %s / %s / %s ===" % (args.backbone, args.tag, variant))
        blob = convert(model_dir, variant, val_gen)
        path = os.path.join(args.out, "%s_%s_%s.tflite" % (args.backbone, args.tag, variant))
        with open(path, "wb") as fh:
            fh.write(blob)
        print("  %s  (%.2f MB)" % (path, len(blob) / 1e6))

        predict = tflite_predictor(path)
        entry = {"path": path, "bytes": len(blob)}
        for split_name, generator in (("val", val_gen), ("test", test_gen)):
            y_true, y_pred, _ = common.predict_generator(predict, generator)
            entry[split_name] = common.evaluate(y_true, y_pred)
            print("  " + common.format_metrics(split_name, entry[split_name]))
        results[variant] = entry

    common.save_json(results, os.path.join(
        "results", "%s_%s_quantisation.json" % (args.backbone, args.tag)))
    print("\nsaved -> results/%s_%s_quantisation.json" % (args.backbone, args.tag))


if __name__ == "__main__":
    main()
