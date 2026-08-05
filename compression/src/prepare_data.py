"""Extract the olive leaf dataset, remove train/test leakage, and build a
stratified, group-aware train/val split.

Uses only packages already present in the course image (numpy / PIL / stdlib).

Why this step exists
--------------------
The published dataset has two kinds of leakage. Both inflate test scores.

1. **Duplicate images** -- the same photograph appears in both train and test.
   Note that **MD5 is not sufficient to detect these**. Of the 100 train/test
   pairs that share a filename, only 19 are byte-identical. The other 81 have
   different MD5s, yet downscaling both to 32x32 gives a mean absolute error of
   **0.0000** at an identical 800x600 source resolution -- they are the same
   photograph, merely re-encoded. This module therefore compares downscaled
   thumbnails (a perceptual hash) rather than byte hashes, so re-encoded copies
   are caught too.

2. **Burst near-duplicates** -- filenames of the form `IMG_YYYYMMDD_HHMMSS.jpg`
   carry a capture timestamp, and those timestamps run in consecutive seconds:
   these are burst shots of the same physical leaf. Neither MD5 nor thumbnail
   matching catches them (the angle shifts slightly), but to a CNN they are
   effectively images it has already seen.

Design decisions
----------------
* **The official test split is left completely intact.** It is the benchmark;
  shrinking it would make results incomparable with other work on this dataset.
* **Removal happens on the train side only**: images duplicated in test, and
  images belonging to a burst that has any test member.
* **The train/val split is group-aware.** Groups are formed by merging burst
  relationships with duplicate relationships (union-find), so every image of a
  given leaf lands on the same side. The split is also stratified by class.
  This replaces the original notebook's `ImageDataGenerator(validation_split=0.3)`,
  which takes the *last 30% of each class's sorted filenames*. Since the filename
  prefix encodes the capture source (`B-*` camera, `IMG_*` phone, `DSC_*` DSLR),
  that produced a validation set systematically shifted away from the training
  distribution.

Outputs
-------
data/prepared/images/{train,test}/<class>/...   extracted images (train de-leaked)
data/prepared/manifest_trainval.csv             file,class,label,split,group
data/prepared/manifest_test.csv                 file,class,label
data/prepared/removed.csv                       removed training images + reason
data/prepared/stats.json                        all counts, for citation in the report
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image

CLASSES = ["Healthy", "aculus_olearius", "olive_peacock_spot"]
LABEL = {c: i for i, c in enumerate(CLASSES)}

# Two photographs taken within this many seconds count as the same burst,
# i.e. the same physical leaf.
BURST_GAP_SECONDS = 10
# Thumbnail edge length. 16x16 greyscale separates distinct photographs while
# tolerating JPEG re-encoding noise.
THUMB_SIZE = 16
# Two thumbnails below this mean absolute error are treated as the same photo.
# Measured: re-encoded copies score 0.0000 and genuinely different photographs
# score far higher, so 0.02 carries a wide safety margin.
DUPLICATE_MAE = 0.02

TIMESTAMP_RE = re.compile(r"IMG_(\d{8})_(\d{6})")


class Entry(object):
    """One image in the archive.

    Identified by (split, cls, filename). Train and test share 100 filenames, so
    keying on (cls, filename) alone would silently collide the two splits.
    """

    __slots__ = ("split", "cls", "filename", "member", "md5", "thumb", "stamp")

    def __init__(self, split, cls, filename, member, md5, thumb, stamp):
        self.split = split
        self.cls = cls
        self.filename = filename
        self.member = member
        self.md5 = md5
        self.thumb = thumb
        self.stamp = stamp

    @property
    def key(self):
        return (self.split, self.cls, self.filename)


class UnionFind(object):
    """Merges burst and duplicate relationships into the groups used for splitting."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_timestamp(filename):
    """Capture time from IMG_YYYYMMDD_HHMMSS.jpg, or None if the name does not match."""
    m = TIMESTAMP_RE.match(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def read_archive(zip_path):
    """Read every image, computing its MD5 and greyscale thumbnail."""
    entries = []
    with zipfile.ZipFile(zip_path) as z:
        members = []
        for member in z.namelist():
            parts = member.split("/")
            # Archive layout is dataset/<split>/<class>/<file>
            if member.endswith("/") or len(parts) < 4:
                continue
            if parts[2] not in CLASSES:
                continue
            members.append((parts[1], parts[2], parts[3], member))

        for i, (split, cls, filename, member) in enumerate(members):
            raw = z.read(member)
            image = Image.open(io.BytesIO(raw)).convert("L").resize(
                (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            thumb = np.asarray(image, dtype=np.float32).ravel() / 255.0
            entries.append(Entry(split, cls, filename, member,
                                 hashlib.md5(raw).hexdigest(), thumb,
                                 parse_timestamp(filename)))
            if (i + 1) % 500 == 0:
                print("    read %d / %d" % (i + 1, len(members)))
    return entries


def find_duplicate_pairs(entries):
    """Find train/test pairs whose thumbnails match below the MAE threshold.

    Compared within each class: a cross-class duplicate would not mean the model
    saw a test image during training for that class, and restricting to one class
    at a time keeps the comparison count manageable.
    """
    pairs = []
    for cls in CLASSES:
        train = [e for e in entries if e.split == "train" and e.cls == cls]
        test = [e for e in entries if e.split == "test" and e.cls == cls]
        if not train or not test:
            continue
        train_matrix = np.stack([e.thumb for e in train])       # (Ntr, D)
        test_matrix = np.stack([e.thumb for e in test])         # (Nte, D)
        # Chunked so we never allocate an Ntr x Nte x D array at once.
        chunk = 64
        for start in range(0, len(test), chunk):
            block = test_matrix[start:start + chunk]
            mae = np.abs(train_matrix[:, None, :] - block[None, :, :]).mean(axis=2)
            for tr_i, te_i in np.argwhere(mae < DUPLICATE_MAE):
                pairs.append((train[tr_i], test[start + te_i], float(mae[tr_i, te_i])))
    return pairs


def build_burst_groups(entries):
    """Chain images of the same class with adjacent timestamps into bursts.

    Returns {entry.key: group_id}. Images without a timestamp each form their own
    group -- there is no way to tell which burst they belong to, so we do not
    merge them with anything.
    """
    groups = {}
    timestamped = collections.defaultdict(list)
    for entry in entries:
        if entry.stamp is None:
            groups[entry.key] = "%s::solo::%s::%s" % (entry.cls, entry.split, entry.filename)
        else:
            timestamped[entry.cls].append(entry)

    for cls, rows in timestamped.items():
        rows.sort(key=lambda e: (e.stamp, e.split, e.filename))
        index = 0
        for i, entry in enumerate(rows):
            if i > 0 and (entry.stamp - rows[i - 1].stamp).total_seconds() > BURST_GAP_SECONDS:
                index += 1
            groups[entry.key] = "%s::burst::%d" % (cls, index)
    return groups


def link_duplicates_within_train(union_find, train_entries):
    """Union training images that duplicate each other.

    Without this, two copies of the same leaf could land on opposite sides of the
    train/val boundary and inflate validation scores the same way test leakage
    inflates test scores.
    """
    by_cls = collections.defaultdict(list)
    for entry in train_entries:
        by_cls[entry.cls].append(entry)

    for rows in by_cls.values():
        if len(rows) < 2:
            continue
        matrix = np.stack([e.thumb for e in rows])
        chunk = 64
        for start in range(0, len(rows), chunk):
            block = matrix[start:start + chunk]
            mae = np.abs(matrix[:, None, :] - block[None, :, :]).mean(axis=2)
            for i, j in np.argwhere(mae < DUPLICATE_MAE):
                if rows[i].key != rows[start + j].key:
                    union_find.union(rows[i].key, rows[start + j].key)


def split_train_val(train_entries, union_find, val_fraction, seed):
    """Stratified by class, grouped by leaf identity.

    Group sizes vary, so within each class we shuffle the groups and take whole
    groups until the target validation count is reached.
    """
    import random

    rng = random.Random(seed)
    assignment = {}
    for cls in CLASSES:
        rows = [e for e in train_entries if e.cls == cls]
        by_group = collections.defaultdict(list)
        for entry in rows:
            by_group[union_find.find(entry.key)].append(entry)

        group_ids = sorted(by_group, key=str)    # sort first so shuffling is reproducible
        rng.shuffle(group_ids)

        target = int(round(len(rows) * val_fraction))
        taken, val_groups = 0, set()
        for gid in group_ids:
            if taken >= target:
                break
            val_groups.add(gid)
            taken += len(by_group[gid])

        for gid, members in by_group.items():
            bucket = "val" if gid in val_groups else "train"
            for entry in members:
                assignment[entry.key] = bucket
    return assignment


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default="/data/olive-leaf-image-dataset.zip",
                        help="path to the original dataset archive")
    parser.add_argument("--out", default="data/prepared", help="output directory")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="fraction of the de-leaked training set held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-leaks", action="store_true",
                        help="keep every leaked image, to build a control run that "
                             "quantifies how much leakage inflates the scores")
    args = parser.parse_args()

    print("Reading archive, computing MD5 and thumbnails ...")
    entries = read_archive(args.zip)
    print("  %d images total" % len(entries))

    print("Matching thumbnails to find train/test duplicates ...")
    dup_pairs = find_duplicate_pairs(entries)
    dup_train_keys = set(tr.key for tr, _te, _mae in dup_pairs)
    md5_identical = sum(1 for tr, te, _mae in dup_pairs if tr.md5 == te.md5)
    print("  %d duplicate pairs covering %d training images "
          "(only %d are MD5-identical; the rest are re-encoded copies)"
          % (len(dup_pairs), len(dup_train_keys), md5_identical))

    groups = build_burst_groups(entries)
    test_bursts = set(groups[e.key] for e in entries if e.split == "test")
    burst_train_keys = set(e.key for e in entries
                           if e.split == "train" and groups[e.key] in test_bursts)
    print("  %d training images share a burst with a test image" % len(burst_train_keys))

    leaked = dup_train_keys | burst_train_keys
    if args.keep_leaks:
        leaked = set()
        print("  --keep-leaks: retaining every leaked image (control run)")
    else:
        print("  removing %d images from the training set" % len(leaked))

    # Groups drive the train/val split: start from bursts, then merge duplicates.
    union_find = UnionFind()
    for entry in entries:
        union_find.union(entry.key, ("burst", groups[entry.key]))

    images_dir = os.path.join(args.out, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    train_entries, test_entries, removed_rows = [], [], []
    print("Extracting images ...")
    with zipfile.ZipFile(args.zip) as z:
        for entry in entries:
            if entry.split == "train" and entry.key in leaked:
                reason = []
                if entry.key in dup_train_keys:
                    reason.append("duplicate_of_test")
                if entry.key in burst_train_keys:
                    reason.append("burst_near_duplicate")
                removed_rows.append((entry.cls, entry.filename, "+".join(reason), entry.md5))
                continue
            target_dir = os.path.join(images_dir, entry.split, entry.cls)
            os.makedirs(target_dir, exist_ok=True)
            with z.open(entry.member) as src, \
                    open(os.path.join(target_dir, entry.filename), "wb") as dst:
                shutil.copyfileobj(src, dst)
            (train_entries if entry.split == "train" else test_entries).append(entry)

    link_duplicates_within_train(union_find, train_entries)

    print("Splitting train/val (stratified by class, grouped by leaf) ...")
    assignment = split_train_val(train_entries, union_find, args.val_fraction, args.seed)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest_trainval.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "class", "label", "split", "group"])
        for entry in sorted(train_entries, key=lambda e: (e.cls, e.filename)):
            writer.writerow([os.path.join("train", entry.cls, entry.filename), entry.cls,
                             LABEL[entry.cls], assignment[entry.key],
                             str(union_find.find(entry.key))])

    with open(os.path.join(args.out, "manifest_test.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "class", "label"])
        for entry in sorted(test_entries, key=lambda e: (e.cls, e.filename)):
            writer.writerow([os.path.join("test", entry.cls, entry.filename),
                             entry.cls, LABEL[entry.cls]])

    with open(os.path.join(args.out, "removed.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "file", "reason", "md5"])
        writer.writerows(sorted(removed_rows))

    original = collections.Counter(e.cls for e in entries if e.split == "train")
    kept = collections.Counter(e.cls for e in train_entries)
    per_split = collections.Counter((e.cls, assignment[e.key]) for e in train_entries)
    test_count = collections.Counter(e.cls for e in test_entries)

    stats = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "burst_gap_seconds": BURST_GAP_SECONDS,
        "duplicate_detection": {
            "method": "%dx%d greyscale thumbnail, mean absolute error < %.3f"
                      % (THUMB_SIZE, THUMB_SIZE, DUPLICATE_MAE),
            "pairs_found": len(dup_pairs),
            "md5_identical_pairs": md5_identical,
            "reencoded_pairs": len(dup_pairs) - md5_identical,
        },
        "leaks_removed": {
            "duplicate_of_test": len(dup_train_keys),
            "burst_near_duplicate": len(burst_train_keys),
            "union": len(leaked),
        },
        "train_original": dict(original),
        "train_after_dedup": dict(kept),
        "train_split": {c: per_split[(c, "train")] for c in CLASSES},
        "val_split": {c: per_split[(c, "val")] for c in CLASSES},
        "test": dict(test_count),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\n%-22s %7s %7s %7s %7s %7s" % ("class", "orig", "dedup", "train", "val", "test"))
    print("-" * 62)
    for cls in CLASSES:
        print("%-22s %7d %7d %7d %7d %7d" % (
            cls, original[cls], kept[cls],
            per_split[(cls, "train")], per_split[(cls, "val")], test_count[cls]))
    print("-" * 62)
    print("%-22s %7d %7d %7d %7d %7d" % (
        "TOTAL", sum(original.values()), sum(kept.values()),
        sum(per_split[(c, "train")] for c in CLASSES),
        sum(per_split[(c, "val")] for c in CLASSES), sum(test_count.values())))
    print("\nWritten to %s" % args.out)


if __name__ == "__main__":
    main()
