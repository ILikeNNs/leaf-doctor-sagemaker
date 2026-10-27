"""SageMaker Processing step: download PlantVillage and build train/val/test ImageFolders.

Source: https://huggingface.co/datasets/mohanty/PlantVillage (CC BY-SA 3.0).
The official color train/test split files are used as-is. The validation set is
carved out of the train split *by leaf*, so photos of the same leaf never end up
in both train and val (PlantVillage has several shots per leaf).
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter

HF_BASE = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main"
WORK_DIR = "/opt/ml/processing/work"  # lives on the job's ML storage volume
OUTPUT_DIR = "/opt/ml/processing/output"


def download(url, dest):
    if os.path.exists(dest):
        return dest
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
    return dest


def read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def leaf_group(rel_path, leaf_map):
    """Same leaf-id logic as the dataset's HF loader script."""
    name = rel_path.split("/")[-1].replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        name = name.replace(ext, "")
    key = name.lower().strip()
    suggestions = leaf_map.get(key, [])
    return suggestions[0] if len(suggestions) == 1 else key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = no cap (use for quick smoke tests)")
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    zip_path = download(f"{HF_BASE}/data.zip", f"{WORK_DIR}/data.zip")
    train_list = read_lines(download(f"{HF_BASE}/splits/color_train.txt", f"{WORK_DIR}/train.txt"))
    test_list = read_lines(download(f"{HF_BASE}/splits/color_test.txt", f"{WORK_DIR}/test.txt"))
    with open(download(f"{HF_BASE}/leaf_grouping/leaf-map.json", f"{WORK_DIR}/leaf-map.json")) as f:
        leaf_map = json.load(f)

    assignment = {}
    for rel in train_list:
        # Deterministic, leaf-grouped split: hash the leaf id, not the file name.
        bucket = int(hashlib.md5(leaf_group(rel, leaf_map).encode()).hexdigest(), 16) % 10_000
        assignment[rel] = "val" if bucket < args.val_fraction * 10_000 else "train"
    for rel in test_list:
        assignment[rel] = "test"

    counts = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            # Zip entries may or may not carry a top-level folder; normalise to "raw/color/...".
            idx = info.filename.find("raw/color/")
            if idx < 0 or info.is_dir():
                continue
            rel = info.filename[idx:]
            split = assignment.get(rel)
            if split is None:
                continue
            class_name = rel.split("/")[2]
            if args.max_per_class and counts[(split, class_name)] >= args.max_per_class:
                continue
            dest_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            with zf.open(info) as src, open(os.path.join(dest_dir, os.path.basename(rel)), "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[(split, class_name)] += 1

    totals = Counter()
    for (split, _), n in counts.items():
        totals[split] += n
    print("Images per split:", dict(totals))
    if set(totals) != {"train", "val", "test"}:
        raise RuntimeError(f"Expected train/val/test images, got {dict(totals)} - check the zip layout.")


if __name__ == "__main__":
    main()
