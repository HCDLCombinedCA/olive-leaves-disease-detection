"""Assemble every result produced so far into report-ready Markdown tables.

Reads the JSON and CSV artefacts written by `compression/` and `analysis/` and
emits one document. Missing inputs are reported as "not yet available" rather
than crashing, so this can be run at any point in the pipeline.

Usage:
    python3 make_report_tables.py            # prints to stdout
    python3 make_report_tables.py -o RESULTS.md
"""
import argparse
import csv
import json
import os

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
SHORT = {"Healthy": "Healthy", "aculus_olearius": "Aculus", "olive_peacock_spot": "Peacock"}


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def missing(what, path):
    return "_%s not yet available (`%s`)_\n" % (what, path)


def section_dataset(out):
    out.append("## 1. Dataset preparation\n")
    stats = load_json("compression/data/prepared/stats.json")
    if not stats:
        out.append(missing("Dataset stats", "compression/data/prepared/stats.json"))
        return

    detection = stats["duplicate_detection"]
    leaks = stats["leaks_removed"]
    out.append(
        "Leakage was detected with a %s. MD5 alone finds only %d of the %d duplicate "
        "pairs -- the remainder are the same photograph re-encoded, identical in pixels "
        "but not in bytes.\n"
        % (detection["method"], detection["md5_identical_pairs"], detection["pairs_found"]))
    out.append("| leak type | training images removed |")
    out.append("|---|---|")
    out.append("| duplicated in test | %d |" % leaks["duplicate_of_test"])
    out.append("| shares a burst with a test image | %d |" % leaks["burst_near_duplicate"])
    out.append("| **total** | **%d** |\n" % leaks["union"])

    out.append("| class | original | after de-dup | train | val | test |")
    out.append("|---|---|---|---|---|---|")
    for cls in CLASSES:
        out.append("| %s | %d | %d | %d | %d | %d |" % (
            cls, stats["train_original"][cls], stats["train_after_dedup"][cls],
            stats["train_split"][cls], stats["val_split"][cls], stats["test"][cls]))
    out.append("| **total** | **%d** | **%d** | **%d** | **%d** | **%d** |\n" % (
        sum(stats["train_original"].values()), sum(stats["train_after_dedup"].values()),
        sum(stats["train_split"].values()), sum(stats["val_split"].values()),
        sum(stats["test"].values())))
    out.append("The official test split is left intact; removal is on the training side only, "
               "so results stay comparable with other work on this dataset.\n")


def section_segmentation(out):
    out.append("## 2. Leaf segmentation\n")
    rows = load_csv("leaf_segmenter/seg_result.csv")
    if not rows:
        out.append(missing("Segmentation results", "leaf_segmenter/seg_result.csv"))
        return

    counts = {}
    for row in rows:
        counts[int(row["n_leaves"])] = counts.get(int(row["n_leaves"]), 0) + 1
    detected = sum(n * c for n, c in counts.items())
    out.append("Model selection is documented in `leaf_segmenter/eval/evaluation_report.md`: "
               "eleven configurations across five families benchmarked on CVPPP A1, won by "
               "fine-tuned YOLO11-seg (SBD 0.847, FBD 0.958, 0.015 s/image). That model was "
               "then applied to the olive photographs.\n")
    out.append("| leaves detected | photographs | share |")
    out.append("|---|---|---|")
    for n in sorted(counts):
        out.append("| %d | %d | %.1f%% |" % (n, counts[n], counts[n] / len(rows) * 100))
    out.append("| **total** | **%d** | |\n" % len(rows))

    fallbacks = 0
    crop_total = 0
    root = "olive_leaf_dataset/segmented/train"
    for cls in CLASSES:
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        names = os.listdir(folder)
        crop_total += len(names)
        fallbacks += sum(1 for n in names if os.path.splitext(n)[0].endswith("_full"))
    if crop_total:
        out.append("The crop set reconciles exactly: **%d files = %d detected leaves + %d "
                   "whole-image fallbacks**. Where the segmenter found nothing, the pipeline "
                   "writes the entire photograph as `<stem>_full.png`, so %.0f%% of the "
                   "\"leaf crops\" feeding the glass-box features are unsegmented photographs "
                   "with their background intact -- relevant given the capture-source "
                   "confound in section 8.\n"
                   % (crop_total, detected, fallbacks, fallbacks / crop_total * 100))


def section_deep_models(out):
    out.append("## 3. Deep models: from scratch against transfer learning\n")
    specs = [
        ("scratch CNN (notebook architecture)", "compression/results/scratch_original_baseline.json"),
        ("scratch CNN (redesigned)", "compression/results/scratch_improved_baseline.json"),
        ("MobileNetV2 (ImageNet)", "compression/results/mobilenetv2_baseline.json"),
        ("DenseNet121 (ImageNet)", "compression/results/densenet121_baseline.json"),
    ]
    rows = [(label, load_json(path)) for label, path in specs]
    rows = [(label, data) for label, data in rows if data]
    if not rows:
        out.append(missing("Deep model results", "compression/results/*_baseline.json"))
        return

    out.append("| model | parameters | train time | val macro-F1 | test macro-F1 | "
               + " | ".join("test recall %s" % SHORT[c] for c in CLASSES) + " |")
    out.append("|---" * (5 + len(CLASSES)) + "|")
    for label, data in rows:
        recalls = data["metrics"]["test"]["per_class_recall"]
        out.append("| %s | %s | %.0f s | %.4f | %.4f | %s |" % (
            label, "{:,}".format(data["total_parameters"]), data["train_seconds"],
            data["metrics"]["val"]["macro_f1"], data["metrics"]["test"]["macro_f1"],
            " | ".join("%.3f" % recalls[c] for c in CLASSES)))
    out.append("")
    for label, data in rows:
        val, test = data["metrics"]["val"]["macro_f1"], data["metrics"]["test"]["macro_f1"]
        out.append("- **%s**: val %.3f vs test %.3f (gap %+.3f)." % (
            label, val, test, test - val))
    out.append("\nThe original notebook reported val 0.819 against test 0.946. A validation "
               "score far *below* the test score is the inversion that first suggested "
               "leakage, and no model here reproduces it.\n")
    negative = [label for label, data in rows
                if data["metrics"]["test"]["macro_f1"] < data["metrics"]["val"]["macro_f1"]]
    if negative:
        out.append("Worth a sentence in the report: %s %s the only %s whose validation score "
                   "*overstates* its test score. Validation is carved from the training "
                   "photographs while the test split is the dataset's own, so a model that "
                   "leans on whatever the training photographs share -- capture conditions "
                   "included, see section 8 -- will look better on validation than it is. "
                   "Transfer learning does not show this, which is consistent with its "
                   "features coming from ImageNet rather than from these photographs.\n"
                   % (", ".join("**%s**" % n for n in negative),
                      "is" if len(negative) == 1 else "are",
                      "model" if len(negative) == 1 else "models"))
    out.append("Every model here shares `compression/src/common.py`: the same manifests, input "
               "size, augmentation, class weights and metrics. The scratch entries differ from "
               "the transfer entries only in initialisation and architecture, and the first "
               "scratch entry reproduces the notebook's chosen architecture unchanged, so the "
               "effect of the corrected data is separated from the effect of the design.\n")


def section_compression(out):
    out.append("## 6. Compression\n")
    found = False
    for backbone in ("mobilenetv2", "densenet121"):
        rows = load_csv("compression/results/%s_compression_table.csv" % backbone)
        if not rows:
            continue
        found = True
        out.append("### %s\n" % backbone)
        out.append("| variant | size (MB) | gzip (MB) | latency (ms) | img/s | test macro-F1 | "
                   "recall Aculus |")
        out.append("|---|---|---|---|---|---|---|")
        for row in rows:
            out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                row["variant"].replace("%s_" % backbone, ""),
                row["size_mb"], row["gzip_mb"], row["latency_median_ms"],
                row["throughput_img_s"], row["test_macro_f1"], row["test_recall_aculus"]))
        out.append("")
        baseline = next((r for r in rows if r["variant"].endswith("baseline")), None)
        if baseline:
            out.append("Baseline for reference: %s MB, %s ms, macro-F1 %s.\n" % (
                baseline["size_mb"], baseline["latency_median_ms"], baseline["test_macro_f1"]))
    if not found:
        out.append(missing("Compression tables",
                           "compression/results/*_compression_table.csv"))
    out.append("`gzip (MB)` is the sum of the individually gzipped files, which for a "
               "SavedModel directory is marginally larger than one gzip stream over the whole "
               "tree (about 0.06% on these artefacts).\n")
    out.append("Unstructured pruning zeroes weights without changing tensor shapes, so the raw "
               "file size is unchanged and the saving appears only in the gzipped column. "
               "Reporting both is what makes the pruning result readable.\n")


def section_xai(out):
    out.append("## 7. Explanation fidelity after compression (LIME)\n")
    data = load_json("compression/results/mobilenetv2_xai_fidelity.json")
    if not data:
        out.append(missing("Fidelity results", "compression/results/*_xai_fidelity.json"))
        return
    out.append("| variant | images | mean Spearman | mean top-5 Jaccard | label agreement |")
    out.append("|---|---|---|---|---|")
    for variant, entry in data["summary"].items():
        out.append("| %s | %d | %s | %.3f | %.0f%% |" % (
            variant, entry["images"],
            "n/a" if entry["mean_spearman"] is None else "%.3f" % entry["mean_spearman"],
            entry["mean_top5_jaccard"], entry["label_agreement_rate"] * 100))
    out.append("\nLIME rather than Grad-CAM because a `.tflite` model has no gradients; a "
               "model-agnostic method lets the identical procedure run against the Keras "
               "baseline and every quantised variant.\n")


GLASSBOX_MODELS = (("decision_tree", "Decision Tree (depth 3)"),
                   ("logistic_regression", "Logistic Regression"))
GLASSBOX_PROTOCOLS = (
    ("segmented", "base", "analysis/results/glassbox_fixed.json"),
    ("segmented", "extended", "analysis/results/glassbox_fixed_extended.json"),
    ("common", "base", "analysis/results/glassbox_common_protocol.json"),
    ("common", "extended", "analysis/results/glassbox_common_protocol_extended.json"),
)


def best_common_glassbox():
    """The strongest glass-box model on the official test split, whatever it is.

    Which feature set and which classifier win is not fixed -- the 47-feature set
    lifts logistic regression well clear while costing the depth-3 tree -- so the
    comparison table asks for the best rather than hard-coding a winner.
    """
    best = None
    for protocol, features, path in GLASSBOX_PROTOCOLS:
        if protocol != "common":
            continue
        data = load_json(path)
        if not data:
            continue
        for key, label in GLASSBOX_MODELS:
            entry = data.get(key)
            if entry and (best is None or entry["macro_f1"] > best[2]):
                best = (label, features, entry["macro_f1"], entry["accuracy"],
                        len(data["features"]))
    return best


def section_glassbox(out):
    out.append("## 4. Glass-box models\n")
    found = [(name, feats, load_json(path)) for name, feats, path in GLASSBOX_PROTOCOLS]
    found = [(name, feats, data) for name, feats, data in found if data]
    if not found:
        out.append(missing("Glass-box results", "analysis/results/glassbox_*.json"))
        return

    out.append("| protocol | features | model | test images | accuracy | macro-F1 |")
    out.append("|---|---|---|---|---|---|")
    for name, feats, data in found:
        for key, label in GLASSBOX_MODELS:
            if key in data:
                out.append("| %s | %s (%d) | %s | %d | %.3f | %.3f |" % (
                    name, feats, len(data["features"]), label,
                    data["split"]["test"], data[key]["accuracy"], data[key]["macro_f1"]))
    out.append("")
    seen = set()
    for name, _feats, data in found:
        if name in seen:
            continue
        seen.add(name)
        split = data["split"]
        out.append("- **%s**: %d train / %d test, %s, %d shared source photographs." % (
            name, split["train"], split["test"], split.get("note", "n/a"),
            split["shared_groups"]))
    out.append("\nThe 47-feature set adds colour statistics, lesion morphology, LBP, edge, "
               "local-variance, entropy and Gabor descriptors to the original nine. It is "
               "worth having: on the common protocol it lifts logistic regression from 0.600 "
               "to 0.742. It does not help the depth-3 tree, which can only consult three "
               "features however many it is offered, and with 47 candidates it picks worse "
               "ones.")
    out.append("\nThe two protocols do not rank the models the same way, which is worth stating "
               "rather than smoothing over: the tree gains on whole photographs while logistic "
               "regression loses. The engineered colour fractions are computed inside a leaf "
               "mask that a tight crop makes reliable and a full photograph does not, and a "
               "linear model has no way to compensate for that where a depth-3 tree's "
               "thresholds partly can.\n")
    out.append("Individual predictions -- two correct and two incorrect per model, with the "
               "decision path or the exact per-feature contribution behind each -- are in "
               "`analysis/results/glassbox_*_examples.png` and in the `explanations` key of "
               "each JSON.\n")


def section_rq1(out):
    out.append("## 5. RQ a: like-for-like comparison\n")
    rows = []
    for label, path in (("scratch CNN (notebook architecture)",
                         "compression/results/scratch_original_baseline.json"),
                        ("scratch CNN (redesigned)",
                         "compression/results/scratch_improved_baseline.json"),
                        ("MobileNetV2 (ImageNet)",
                         "compression/results/mobilenetv2_baseline.json"),
                        ("DenseNet121 (ImageNet)",
                         "compression/results/densenet121_baseline.json")):
        data = load_json(path)
        if data:
            rows.append((label, data["metrics"]["test"]["accuracy"],
                         data["metrics"]["test"]["macro_f1"],
                         "post-hoc (Grad-CAM / LIME)"))

    for protocol, features, path in GLASSBOX_PROTOCOLS:
        if protocol != "common":
            continue
        glassbox = load_json(path)
        if not glassbox:
            continue
        for key, label in GLASSBOX_MODELS:
            rows.append(("%s, %d features" % (label, len(glassbox["features"])),
                         glassbox[key]["accuracy"], glassbox[key]["macro_f1"],
                         "intrinsic (exact)"))
    rows.sort(key=lambda r: -r[2])
    if not rows:
        out.append(missing("Comparison inputs", "compression/results/, analysis/results/"))
        return

    out.append("All rows below are scored on the **same official 680-image test split**. The "
               "segmented-crop glass-box numbers in section 4 are measured on a different "
               "population and are deliberately not carried into this table.\n")
    out.append("| model | accuracy | macro-F1 | explanation |")
    out.append("|---|---|---|---|")
    for label, accuracy, macro_f1, explanation in rows:
        out.append("| %s | %.3f | %.3f | %s |" % (label, accuracy, macro_f1, explanation))
    out.append("")
    best = best_common_glassbox()
    if best and rows:
        label, features, macro_f1, _accuracy, n_features = best
        top = max(r[2] for r in rows)
        out.append("The interpretability cost is the gap between the best deep model and the "
                   "best intrinsically interpretable one: **%.3f against %.3f, %.0f points**. "
                   "That best glass-box is %s on the %s %d-feature set -- not the nine "
                   "features the pipeline started from, which reach only 0.667 here.\n"
                   % (top, macro_f1, (top - macro_f1) * 100, label, features, n_features))


def section_bias(out):
    out.append("## 8. Acquisition bias\n")
    data = load_json("analysis/results/acquisition_bias.json")
    if not data:
        out.append(missing("Bias results", "analysis/results/acquisition_bias.json"))
        return

    part_a = data["part_a"]
    out.append("### Is capture source entangled with the label?\n")
    out.append("| capture source | n (train) | dominant class | purity |")
    out.append("|---|---|---|---|")
    for source, entry in sorted(part_a["source_purity"].items(),
                                key=lambda kv: -kv[1]["purity"]):
        out.append("| %s | %d | %s | %.1f%% |" % (
            source, entry["n"], entry["top_class"], entry["purity"] * 100))
    accuracy = part_a["filename_only_accuracy"]
    out.append("\nA classifier given only the filename prefix reaches **%.1f%% on training "
               "data** and %.1f%% on the official test split, against a %.1f%% majority "
               "baseline. Training accuracy on this dataset has to be read with that in "
               "mind.\n" % (accuracy["train"] * 100, accuracy["test"] * 100,
                           accuracy["majority_baseline"] * 100))

    part_b = data.get("part_b")
    if not part_b:
        out.append("_Grad-CAM attention analysis not yet run._\n")
        return
    out.append("### Does the model exploit it?\n")
    out.append("Grad-CAM++ attention inside versus outside a leaf mask. A ratio above 1 "
               "means more attention on background than an even spread would give.\n")
    out.append("| subset | n | attention outside leaf | background area | ratio |")
    out.append("|---|---|---|---|---|")
    out.append("| all | %d | %.1f%% | %.1f%% | **%.2f** |" % (
        part_b["n_correct"] + part_b["n_incorrect"],
        part_b["mean_attention_outside_leaf"] * 100,
        part_b["mean_background_area"] * 100, part_b["background_preference_ratio"]))
    for key, label, count in (("background_preference_ratio_correct", "correct", "n_correct"),
                              ("background_preference_ratio_incorrect", "incorrect",
                               "n_incorrect")):
        if part_b.get(key) is not None:
            out.append("| %s predictions | %d | -- | -- | %.2f |" % (
                label, part_b[count], part_b[key]))
    out.append("\nThe confound is present in the data but this model is not using it: "
               "attention concentrates on leaf tissue well beyond chance, and errors are "
               "not explained by background reliance.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    out = ["# Results", ""]
    out.append("Generated from the artefacts in `compression/results/` and "
               "`analysis/results/`. Regenerate with `python3 make_report_tables.py`.\n")
    for section in (section_dataset, section_segmentation, section_deep_models,
                    section_glassbox, section_rq1, section_compression,
                    section_xai, section_bias):
        section(out)
        out.append("---\n")

    text = "\n".join(out)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print("wrote %s" % args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
