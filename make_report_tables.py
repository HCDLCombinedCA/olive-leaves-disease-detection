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


def section_baselines(out):
    out.append("## 2. Baseline models\n")
    rows = []
    for backbone in ("mobilenetv2", "densenet121"):
        data = load_json("compression/results/%s_baseline.json" % backbone)
        if data:
            rows.append((backbone, data))
    if not rows:
        out.append(missing("Baseline results", "compression/results/*_baseline.json"))
        return

    out.append("| backbone | parameters | train time | val macro-F1 | test macro-F1 | "
               + " | ".join("test recall %s" % SHORT[c] for c in CLASSES) + " |")
    out.append("|---" * (5 + len(CLASSES)) + "|")
    for backbone, data in rows:
        recalls = data["metrics"]["test"]["per_class_recall"]
        out.append("| %s | %s | %.0f s | %.4f | %.4f | %s |" % (
            backbone, "{:,}".format(data["total_parameters"]), data["train_seconds"],
            data["metrics"]["val"]["macro_f1"], data["metrics"]["test"]["macro_f1"],
            " | ".join("%.3f" % recalls[c] for c in CLASSES)))
    out.append("")
    for backbone, data in rows:
        val, test = data["metrics"]["val"]["macro_f1"], data["metrics"]["test"]["macro_f1"]
        out.append("- **%s**: val %.3f vs test %.3f (gap %+.3f)." % (
            backbone, val, test, test - val))
    out.append("\nA small gap in this direction is what a clean split should give. The original "
               "notebook reported val 0.819 against test 0.946 -- the inversion that first "
               "suggested leakage.\n")


def section_compression(out):
    out.append("## 3. Compression\n")
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
    out.append("Unstructured pruning zeroes weights without changing tensor shapes, so the raw "
               "file size is unchanged and the saving appears only in the gzipped column. "
               "Reporting both is what makes the pruning result readable.\n")


def section_xai(out):
    out.append("## 4. Explanation fidelity after compression (LIME)\n")
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


def section_glassbox(out):
    out.append("## 5. Glass-box models\n")
    data = load_json("analysis/results/glassbox_fixed.json")
    if not data:
        out.append(missing("Glass-box results", "analysis/results/glassbox_fixed.json"))
        return
    out.append("| model | accuracy | macro-F1 |")
    out.append("|---|---|---|")
    for key, label in (("decision_tree", "Decision Tree (depth 3)"),
                       ("logistic_regression", "Logistic Regression")):
        if key in data:
            out.append("| %s | %.3f | %.3f |" % (
                label, data[key]["accuracy"], data[key]["macro_f1"]))
    split = data.get("split", {})
    out.append("\nSplit: %d train / %d test crops, grouped by source photograph "
               "(%d shared groups).\n" % (split.get("train", 0), split.get("test", 0),
                                          split.get("shared_groups", -1)))


def section_bias(out):
    out.append("## 6. Acquisition bias\n")
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
    for section in (section_dataset, section_baselines, section_compression,
                    section_xai, section_glassbox, section_bias):
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
