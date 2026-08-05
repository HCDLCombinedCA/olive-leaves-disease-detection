"""Dataset audit: the data-quality checks the assignment plan asks for.

`plan.txt` items 10 and 11 list six checks. Two were already done elsewhere and
are summarised here for completeness; the other four are computed by this script.

  done elsewhere:
    duplicates / near-duplicates   compression/src/prepare_data.py
    background and camera bias     analysis/acquisition_bias.py

  computed here:
    corrupt or unreadable files
    image dimensions and file formats
    class balance
    colour and intensity distribution per class

Everything reads the original archive, so the audit describes the dataset as
published rather than after de-duplication.

Usage:
    ./run.sh python dataset_audit.py
"""
import argparse
import collections
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
# Coarse hue bands, in degrees, chosen to match the vocabulary the glass-box
# feature extractor already uses so the two descriptions agree.
HUE_BANDS = [("green", 70, 160), ("yellow", 40, 70), ("orange_brown", 10, 40)]
SEED = 42


def capture_source(filename):
    """Group a filename by the device that produced it, as in acquisition_bias.py."""
    import re
    lower = filename.lower()
    if lower.startswith("img_"):
        return "IMG_"
    if lower.startswith("dsc_"):
        return "DSC_"
    if lower.startswith("b"):
        return "B*"
    if re.match(r"^a[-\d]", lower):
        return "A*"
    if re.match(r"^\d", lower):
        return "numeric"
    return "other"


def audit(zip_path, sample_per_class):
    records = []
    corrupt = []
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for name in archive.namelist():
            parts = name.split("/")
            if name.endswith("/") or len(parts) < 4 or parts[2] not in CLASSES:
                continue
            members.append((parts[1], parts[2], parts[3], name))

        # Which members get decoded for colour statistics. Sampling must be
        # random within each class, not "the first N in archive order": the
        # archive is grouped by capture source, so taking a prefix returns one
        # camera rather than a cross-section of the class.
        import random
        rng = random.Random(SEED)
        by_class = collections.defaultdict(list)
        for entry in members:
            by_class[entry[1]].append(entry[3])
        colour_sample = set()
        for cls, names in by_class.items():
            names = sorted(names)
            rng.shuffle(names)
            colour_sample.update(names[:sample_per_class])

        for split, cls, filename, member in members:
            raw = archive.read(member)
            try:
                image = Image.open(io.BytesIO(raw))
                image.verify()                      # structural check
                image = Image.open(io.BytesIO(raw)) # verify() exhausts the file
                width, height = image.size
                fmt = image.format
                mode = image.mode
            except Exception as exc:                # noqa: BLE001 - report anything
                corrupt.append({"split": split, "class": cls, "file": filename,
                                "error": "%s: %s" % (type(exc).__name__, exc)})
                continue

            record = {"split": split, "class": cls, "file": filename,
                      "source": capture_source(filename),
                      "width": width, "height": height, "format": fmt, "mode": mode,
                      "bytes": len(raw)}

            if member in colour_sample:
                small = image.convert("RGB").resize((64, 64), Image.LANCZOS)
                array = np.asarray(small, dtype=np.float32) / 255.0
                import colorsys
                # Vectorised HSV is not in PIL; convert via numpy on the small image.
                maxc = array.max(axis=2)
                minc = array.min(axis=2)
                value = maxc
                saturation = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-8), 0)
                # Hue in degrees
                rc, gc, bc = array[..., 0], array[..., 1], array[..., 2]
                delta = np.maximum(maxc - minc, 1e-8)
                hue = np.zeros_like(maxc)
                mask = maxc == rc
                hue[mask] = ((gc - bc)[mask] / delta[mask]) % 6
                mask = (maxc == gc) & (maxc != rc)
                hue[mask] = ((bc - rc)[mask] / delta[mask]) + 2
                mask = (maxc == bc) & (maxc != rc) & (maxc != gc)
                hue[mask] = ((rc - gc)[mask] / delta[mask]) + 4
                hue = (hue * 60) % 360
                del colorsys

                record["mean_intensity"] = float(value.mean())
                record["std_intensity"] = float(value.std())
                record["mean_saturation"] = float(saturation.mean())
                for band, low, high in HUE_BANDS:
                    lit = (saturation > 0.2) & (value > 0.15)
                    record["frac_" + band] = float(
                        (((hue >= low) & (hue < high)) & lit).mean())
            records.append(record)
    return records, corrupt


def summarise(records, corrupt):
    out = {}

    out["counts"] = {}
    for split in ("train", "test"):
        counts = collections.Counter(r["class"] for r in records if r["split"] == split)
        out["counts"][split] = {c: counts[c] for c in CLASSES}
    total = sum(sum(v.values()) for v in out["counts"].values())
    out["counts"]["total"] = total

    out["corrupt"] = {"count": len(corrupt), "files": corrupt[:20]}

    dims = collections.Counter((r["width"], r["height"]) for r in records)
    out["dimensions"] = {
        "distinct": len(dims),
        "most_common": [{"size": "%dx%d" % k, "n": v, "share": round(v / len(records), 4)}
                        for k, v in dims.most_common(8)],
    }
    out["formats"] = dict(collections.Counter(r["format"] for r in records))
    out["modes"] = dict(collections.Counter(r["mode"] for r in records))

    sizes = np.array([r["bytes"] for r in records], dtype=np.float64)
    out["file_size_kb"] = {
        "min": round(sizes.min() / 1024, 1), "median": round(float(np.median(sizes)) / 1024, 1),
        "max": round(sizes.max() / 1024, 1),
    }

    colour = {}
    for cls in CLASSES:
        subset = [r for r in records if r["class"] == cls and "mean_intensity" in r]
        if not subset:
            continue
        entry = {"n_sampled": len(subset)}
        for key in ("mean_intensity", "std_intensity", "mean_saturation"):
            entry[key] = round(float(np.mean([r[key] for r in subset])), 4)
        for band, _lo, _hi in HUE_BANDS:
            entry["frac_" + band] = round(
                float(np.mean([r["frac_" + band] for r in subset])), 4)
        colour[cls] = entry
    out["colour"] = colour

    # The same statistics split by capture source. If a colour feature tracks the
    # camera rather than the class, that shows up here and nowhere else.
    by_source = {}
    for cls in CLASSES:
        for source in sorted({r["source"] for r in records}):
            subset = [r for r in records
                      if r["class"] == cls and r["source"] == source
                      and "mean_intensity" in r]
            if len(subset) < 10:
                continue
            by_source["%s / %s" % (source, cls)] = {
                "n_sampled": len(subset),
                "mean_intensity": round(float(np.mean([r["mean_intensity"] for r in subset])), 4),
                "frac_orange_brown": round(
                    float(np.mean([r["frac_orange_brown"] for r in subset])), 4),
                "frac_green": round(float(np.mean([r["frac_green"] for r in subset])), 4),
            }
    out["colour_by_source"] = by_source
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="../olive-leaf-image-dataset.zip")
    parser.add_argument("--sample-per-class", type=int, default=400,
                        help="images per class decoded for the colour statistics")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    print("Auditing %s ..." % args.zip)
    records, corrupt = audit(args.zip, args.sample_per_class)
    summary = summarise(records, corrupt)

    print("\n=== counts ===")
    for split in ("train", "test"):
        row = summary["counts"][split]
        print("  %-6s %s  (total %d)" % (split, row, sum(row.values())))

    print("\n=== integrity ===")
    print("  corrupt or unreadable files: %d" % summary["corrupt"]["count"])
    print("  formats: %s" % summary["formats"])
    print("  colour modes: %s" % summary["modes"])

    print("\n=== dimensions ===")
    print("  %d distinct sizes" % summary["dimensions"]["distinct"])
    for entry in summary["dimensions"]["most_common"]:
        print("    %-12s %5d  (%.1f%%)" % (entry["size"], entry["n"], entry["share"] * 100))
    print("  file size KB: %s" % summary["file_size_kb"])

    print("\n=== colour and intensity by class (sampled) ===")
    header = "%-22s %8s %10s %8s %8s %8s %8s" % (
        "class", "n", "intensity", "sat", "green", "yellow", "brown")
    print("  " + header)
    for cls, entry in summary["colour"].items():
        print("  %-22s %8d %10.3f %8.3f %8.3f %8.3f %8.3f" % (
            cls, entry["n_sampled"], entry["mean_intensity"], entry["mean_saturation"],
            entry["frac_green"], entry["frac_yellow"], entry["frac_orange_brown"]))

    print("\n=== colour by capture source (groups with >=10 sampled) ===")
    print("  %-30s %8s %10s %10s" % ("source / class", "n", "intensity", "brown"))
    for key, entry in sorted(summary["colour_by_source"].items()):
        print("  %-30s %8d %10.3f %10.3f" % (
            key, entry["n_sampled"], entry["mean_intensity"], entry["frac_orange_brown"]))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dataset_audit.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
