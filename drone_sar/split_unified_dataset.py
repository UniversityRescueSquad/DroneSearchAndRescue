"""Reorganize the flat unified dataset into train/val/test subdirectories.

The harmonization notebook (notebooks/dataset_demo_and_harmonization.ipynb)
writes all images into data/unified_dataset/ with labels in
data/unified_dataset/labels/. The original splits are encoded in the
filenames (sard_train_*, sard_valid_*, sard_test_*, heridal_train_*,
heridal_test_*). This script moves each image and its XML label into
<root>/<split>/ and <root>/<split>/labels/ so each split can be loaded
with IPSARDataset(<root>/<split>).

HERIDAL has no validation split of its own, so the tail of each location
category's numbered sequence is carved out of heridal_train for val.
A contiguous tail (not a random sample) is used because consecutive shot
numbers within a category are often near-duplicates.
"""

import argparse
import os
import re
import shutil
from collections import Counter, defaultdict

HERIDAL_VAL_FRACTION = 0.1
HERIDAL_TRAIN_RE = re.compile(r"^heridal_train_train_(?P<category>[A-Za-z]+)_(?P<num>\d+)\.")


def assign_splits(image_names: list[str]) -> dict[str, str]:
    """Map each image filename to its split (train/val/test)."""
    splits = {}
    heridal_train_by_category = defaultdict(list)

    for name in image_names:
        if name.startswith("sard_train_"):
            splits[name] = "train"
        elif name.startswith("sard_valid_"):
            splits[name] = "val"
        elif name.startswith("sard_test_") or name.startswith("heridal_test_"):
            splits[name] = "test"
        else:
            match = HERIDAL_TRAIN_RE.match(name)
            if not match:
                raise ValueError(f"Unrecognized filename pattern: {name}")
            heridal_train_by_category[match["category"]].append(
                (int(match["num"]), name)
            )

    for category, numbered in heridal_train_by_category.items():
        numbered.sort()
        n_val = max(1, round(len(numbered) * HERIDAL_VAL_FRACTION))
        for i, (_, name) in enumerate(numbered):
            splits[name] = "val" if i >= len(numbered) - n_val else "train"

    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/unified_dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels_dir = os.path.join(args.root, "labels")
    image_names = [
        f for f in os.listdir(args.root)
        if os.path.isfile(os.path.join(args.root, f))
    ]
    if not image_names:
        print(f"No images left in {args.root} root - nothing to do.")
        return

    splits = assign_splits(image_names)

    missing_labels = []
    for name in image_names:
        label_name = os.path.splitext(name)[0] + ".xml"
        if not os.path.exists(os.path.join(labels_dir, label_name)):
            missing_labels.append(name)
    if missing_labels:
        raise FileNotFoundError(
            f"{len(missing_labels)} images have no XML label, e.g. {missing_labels[:3]}"
        )

    counts = Counter()
    for name, split in splits.items():
        counts[split, name.split("_")[0]] += 1
        if args.dry_run:
            continue
        split_dir = os.path.join(args.root, split)
        split_labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(split_labels_dir, exist_ok=True)
        label_name = os.path.splitext(name)[0] + ".xml"
        shutil.move(os.path.join(args.root, name), os.path.join(split_dir, name))
        shutil.move(
            os.path.join(labels_dir, label_name),
            os.path.join(split_labels_dir, label_name),
        )

    for split in ("train", "val", "test"):
        total = sum(n for (s, _), n in counts.items() if s == split)
        by_source = {src: n for (s, src), n in counts.items() if s == split}
        print(f"{split}: {total} images {by_source}")
    if args.dry_run:
        print("Dry run - no files moved.")


if __name__ == "__main__":
    main()
