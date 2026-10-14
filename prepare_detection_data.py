"""
Export BDD100K's box labels as a YOLO dataset for fine-tuning the detector, without the
614 test images, so that evaluating on them afterwards stays a fair comparison.

- Every dataset image whose file content equals a test image is left out (duplicates
  too), matched by MD5 rather than by file name.
- The remaining images are split at random, with a fixed seed, into train and val. Val
  picks the best epoch and shows overfitting during training; the test images are never
  used for training or for any choice.
- Classes: the 10 standard BDD100K detection classes. Rare extra labels (other vehicle,
  other person, trailer) are dropped and counted.
- Writes bdd100k_det/: images/{train,val}, labels/{train,val}, data.yaml, split.json.

Run:  python prepare_detection_data.py [--val-fraction 0.1] [--seed 42] [--overwrite]
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter

from evaluate_detection import BDD_CLASSES, TEST_IMG_DIR, file_md5, normalize_label

OUT_DIR = "bdd100k_det"


def yolo_line(class_id, box):
    """
    Return one YOLO label line for a box given as FiftyOne gives it, [x, y, w, h] relative
    to the image with (x, y) the top-left corner; None if nothing is left after clipping
    the box to the image.
    """
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def excluded_indices(dataset_paths, test_paths):
    """Return the indices of dataset files whose bytes equal some test file's bytes."""
    sizes = {os.path.getsize(p) for p in test_paths}
    digests = {file_md5(p) for p in test_paths}
    return {i for i, p in enumerate(dataset_paths)
            if os.path.getsize(p) in sizes and file_md5(p) in digests}


def split_indices(indices, val_fraction, seed):
    """Shuffle indices with a fixed seed and return (train, val), each sorted."""
    order = sorted(indices)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])


def write_split(out_dir, split, indices, filepaths, labels, boxes):
    """Copy a split's images and write its label files. Returns the split's statistics."""
    image_dir = os.path.join(out_dir, "images", split)
    label_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(image_dir)
    os.makedirs(label_dir)
    class_id = {name: k for k, name in enumerate(BDD_CLASSES)}
    boxes_per_class, dropped = Counter(), Counter()
    used_names, empty = set(), 0
    for n, i in enumerate(indices, start=1):
        base, ext = os.path.splitext(os.path.basename(filepaths[i]))
        name = base if base not in used_names else f"{base}_{i}"
        used_names.add(name)
        shutil.copyfile(filepaths[i], os.path.join(image_dir, name + ext))
        lines = []
        for label, box in zip(labels[i] or [], boxes[i] or []):
            label = normalize_label(label)
            if label not in class_id:
                dropped[label] += 1
                continue
            line = yolo_line(class_id[label], box)
            if line is not None:
                lines.append(line)
                boxes_per_class[label] += 1
        empty += not lines
        with open(os.path.join(label_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if n % 1000 == 0:
            print(f"  [{split}] {n}/{len(indices)}")
    return {"images": len(indices), "images_without_boxes": empty,
            "boxes": dict(boxes_per_class), "dropped_labels": dict(dropped)}


def write_data_yaml(out_dir):
    """Write the Ultralytics dataset file (absolute path; train_detector.py refreshes it)."""
    names = "\n".join(f"  {k}: \"{name}\"" for k, name in enumerate(BDD_CLASSES))
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")


def prepare(filepaths, labels, boxes, test_paths, out_dir, val_fraction, seed):
    """Build the dataset from the BDD100K fields; returns the summary written to split.json."""
    print("Finding the test images in the dataset by file content...")
    excluded = excluded_indices(filepaths, test_paths)
    print(f"  {len(excluded)} dataset images equal a test image and are left out "
          f"({len(test_paths)} test images)")
    if len(excluded) < len(test_paths):
        sys.exit("Some test images were not found in the dataset; stopping so nothing leaks unnoticed.")
    remaining = [i for i in range(len(filepaths)) if i not in excluded]
    train, val = split_indices(remaining, val_fraction, seed)
    print(f"  train {len(train)}, val {len(val)}")

    os.makedirs(out_dir)
    summary = {"seed": seed, "val_fraction": val_fraction, "classes": BDD_CLASSES,
               "excluded_test_images": len(excluded), "test_images": len(test_paths)}
    for split, indices in [("train", train), ("val", val)]:
        print(f"Writing {split}...")
        summary[split] = write_split(out_dir, split, indices, filepaths, labels, boxes)
    # fingerprint of the excluded files, to check later that no test image is in train/val
    summary["excluded_md5"] = sorted(file_md5(filepaths[i]) for i in excluded)
    write_data_yaml(out_dir)
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help=f"delete and rebuild {OUT_DIR}/")
    args = parser.parse_args()

    if os.path.exists(OUT_DIR):
        if not args.overwrite:
            sys.exit(f"{OUT_DIR}/ already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(OUT_DIR)

    import fiftyone.utils.huggingface as fouh

    test_paths = [os.path.join(TEST_IMG_DIR, f) for f in sorted(os.listdir(TEST_IMG_DIR))]
    print("Loading BDD100K (local Hugging Face cache)...")
    dataset = fouh.load_from_hub("dgural/bdd100k")
    summary = prepare(dataset.values("filepath"), dataset.values("detections.detections.label"),
                      dataset.values("detections.detections.bounding_box"), test_paths,
                      OUT_DIR, args.val_fraction, args.seed)

    print("\nBoxes per class (train / val):")
    for name in BDD_CLASSES:
        print(f"  {name:15}{summary['train']['boxes'].get(name, 0):8}{summary['val']['boxes'].get(name, 0):7}")
    dropped = Counter(summary["train"]["dropped_labels"]) + Counter(summary["val"]["dropped_labels"])
    if dropped:
        print("Dropped (not one of the 10 classes): " + ", ".join(f"{k} {v}" for k, v in dropped.most_common()))
    print(f"\nWrote {OUT_DIR}/ (data.yaml, split.json). Next: python train_detector.py")


if __name__ == "__main__":
    main()
