"""
Evaluate YOLOv8 object detection on the same 614 held-out test images as the
segmentation model, against BDD100K's own box labels.

Works with the stock COCO weights and with a model fine-tuned on BDD100K
(train_detector.py). The stock model's classes are mapped to BDD100K's; COCO has no
"traffic sign" and no "rider" class, so it cannot detect those two at all: they are left
out of its mAP and reported as the share of labelled objects it cannot find. For a fair
before/after comparison the script also reports the mAP over the 8 classes both models
share.

The test images are found in the BDD100K dataset by file content (MD5), so the result
does not depend on reproducing the shuffle that made the split.

Scoring uses pycocotools, the standard COCO evaluation: mAP over IoU 0.50:0.95, mAP at
IoU 0.50, per class, and split by time of day.

Needs:  pip install pycocotools
Run:    python evaluate_detection.py [--model yolov8s.pt] [--imgsz 640] [--results detection_results.json]
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

DATA_ROOT = "bdd100k_drivable_subset"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
METADATA_PATH = os.path.join(DATA_ROOT, "test", "metadata.json")

# The 10 standard BDD100K detection classes
BDD_CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle", "traffic light", "traffic sign"]
# The 8 of them the stock COCO model can output (COCO's "person" counts as pedestrian)
COCO_SHARED = [c for c in BDD_CLASSES if c not in ("rider", "traffic sign")]


def normalize_label(label):
    """BDD100K versions and COCO call pedestrians 'person'; use 'pedestrian' throughout."""
    return "pedestrian" if label == "person" else label


def bdd_mapping(model_names):
    """
    Return {model class index: BDD100K class} for the model's classes that are BDD100K
    classes. For the stock COCO model that is the 8 shared classes (its other classes are
    ignored); for a model fine-tuned on BDD100K it is all 10.
    """
    mapping = {}
    for idx, name in model_names.items():
        name = normalize_label(name)
        if name in BDD_CLASSES:
            mapping[int(idx)] = name
    return mapping


def shared_means(result):
    """mAP and mAP50 over the COCO-shared classes present in the images (for before/after)."""
    rows = [r for name, r in result["per_class"].items() if name in COCO_SHARED]
    if not rows:
        return 0.0, 0.0
    return float(np.mean([r["AP"] for r in rows])), float(np.mean([r["AP50"] for r in rows]))


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def match_by_content(test_paths, dataset_paths):
    """Return {test path: index into dataset_paths} for files with identical bytes."""
    by_size = defaultdict(list)
    for i, path in enumerate(dataset_paths):
        by_size[os.path.getsize(path)].append(i)
    matches = {}
    for test_path in test_paths:
        candidates = by_size.get(os.path.getsize(test_path), [])
        if not candidates:
            continue
        digest = file_md5(test_path)
        for i in candidates:
            if file_md5(dataset_paths[i]) == digest:
                matches[test_path] = i
                break
    return matches


def build_coco_gt(images, class_names):
    """
    Build a COCO ground-truth dict.

    images: list of dicts with id, file_name, width, height, and 'boxes': a list of
    (label, [x, y, w, h]) with the box relative to the image size (FiftyOne's format).
    Boxes whose label is not in class_names are left out.
    """
    categories = [{"id": k + 1, "name": name} for k, name in enumerate(class_names)]
    cat_id = {name: k + 1 for k, name in enumerate(class_names)}
    annotations = []
    for image in images:
        w_img, h_img = image["width"], image["height"]
        for label, (x, y, w, h) in image["boxes"]:
            if label not in cat_id:
                continue
            box = [x * w_img, y * h_img, w * w_img, h * h_img]
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image["id"],
                "category_id": cat_id[label],
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            })
    return {
        "info": {},
        "licenses": [],
        "images": [{"id": im["id"], "file_name": im["file_name"],
                    "width": im["width"], "height": im["height"]} for im in images],
        "annotations": annotations,
        "categories": categories,
    }


def coco_evaluate(gt_dict, predictions, image_ids=None, verbose=False):
    """
    Score predictions (COCO result dicts: image_id, category_id, bbox in pixels as
    [x, y, w, h], score) with pycocotools. Returns mAP, mAP50 and per-class AP / AP50 /
    number of ground-truth boxes, restricted to image_ids when given.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = sorted(image_ids if image_ids is not None else [im["id"] for im in gt_dict["images"]])
    keep = set(image_ids)
    n_gt = Counter(a["category_id"] for a in gt_dict["annotations"] if a["image_id"] in keep)
    names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    preds = [p for p in predictions if p["image_id"] in keep]
    if not preds:
        return {"mAP": 0.0, "mAP50": 0.0, "per_class": {
            names[c]: {"AP": 0.0, "AP50": 0.0, "gt_boxes": n_gt[c]} for c in names if n_gt[c]}}

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(preds)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        evaluator.summarize()
    if verbose:
        print(summary.getvalue())

    precision = evaluator.eval["precision"]  # [IoU thresholds, recall, class, area, max dets]
    per_class = {}
    for k, cat in enumerate(evaluator.params.catIds):
        if n_gt[cat] == 0:
            continue  # no labelled objects of this class in these images
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_class[names[cat]] = {
            "AP": float(np.mean(p_all[p_all > -1])) if np.any(p_all > -1) else 0.0,
            "AP50": float(np.mean(p_50[p_50 > -1])) if np.any(p_50 > -1) else 0.0,
            "gt_boxes": n_gt[cat],
        }
    return {"mAP": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1]), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 weights (default: the pipeline's yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (default 640, as in pipeline.py)")
    parser.add_argument("--results", default="detection_results.json", help="where to save the numbers")
    args = parser.parse_args()

    import torch
    import fiftyone.utils.huggingface as fouh
    from ultralytics import YOLO

    model = YOLO(args.model)
    mapping = bdd_mapping(model.names)
    class_names = sorted(set(mapping.values()))
    cannot_output = [c for c in BDD_CLASSES if c not in class_names]

    # --- Find the test images in BDD100K by content ---
    test_files = sorted(os.listdir(TEST_IMG_DIR))
    test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]
    print(f"{len(test_paths)} test images in {TEST_IMG_DIR}")

    print("Loading BDD100K (local Hugging Face cache)...")
    dataset = fouh.load_from_hub("dgural/bdd100k")
    filepaths = dataset.values("filepath")
    labels = [[normalize_label(x) for x in (ls or [])] for ls in dataset.values("detections.detections.label")]
    boxes = dataset.values("detections.detections.bounding_box")
    timeofday = dataset.values("timeofday.label")
    weather = dataset.values("weather.label")

    print("Matching test images to dataset samples by file content...")
    matches = match_by_content(test_paths, filepaths)
    print(f"  matched {len(matches)}/{len(test_paths)}")
    if len(matches) < len(test_paths):
        print("  unmatched images are left out of the evaluation")

    # Side check: does test/metadata.json (used for the segmentation breakdown) agree?
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = {m["filename"]: m for m in json.load(f)}
        agree = sum(
            1 for fname, path in zip(test_files, test_paths) if path in matches
            and metadata.get(fname, {}).get("timeofday") == timeofday[matches[path]]
            and metadata.get(fname, {}).get("weather") == weather[matches[path]]
        )
        print(f"  metadata.json agrees with the dataset on {agree}/{len(matches)} images")

    # --- Ground truth ---
    all_labels = Counter()
    for path, i in matches.items():
        all_labels.update(labels[i])

    print("\nLabelled objects in the test images:")
    for name, count in all_labels.most_common():
        note = ("" if name in class_names
                else "  <- the model has no such class" if name in cannot_output else "  <- not evaluated")
        print(f"  {name:15} {count:6}{note}")
    total = sum(all_labels.values())
    missing = sum(all_labels[n] for n in cannot_output)
    if cannot_output:
        print(f"  {missing}/{total} labelled objects ({100 * missing / max(total, 1):.1f}%) are classes "
              f"this model cannot output ({', '.join(cannot_output)})")

    images = []
    image_ids = {}
    for image_id, (fname, path) in enumerate(zip(test_files, test_paths), start=1):
        if path not in matches:
            continue
        i = matches[path]
        width, height = Image.open(path).size
        images.append({"id": image_id, "file_name": fname, "width": width, "height": height,
                       "boxes": list(zip(labels[i], boxes[i] or []))})
        image_ids[path] = image_id
    gt_dict = build_coco_gt(images, class_names)
    cat_id = {c["name"]: c["id"] for c in gt_dict["categories"]}

    # --- Predictions ---
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nRunning {args.model} at imgsz={args.imgsz} on {'GPU' if device == 0 else 'CPU'}...")
    predictions = []
    for n, (path, image_id) in enumerate(image_ids.items(), start=1):
        # conf=0.001 and iou=0.7 are Ultralytics' own settings for computing mAP
        result = model.predict(path, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                               device=device, verbose=False)[0]
        for (x1, y1, x2, y2), score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(),
                                                result.boxes.cls.cpu().numpy()):
            bdd = mapping.get(int(cls))
            if bdd is None:
                continue
            predictions.append({"image_id": image_id, "category_id": cat_id[bdd],
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(score)})
        if n % 100 == 0:
            print(f"  {n}/{len(image_ids)}")

    # --- Scores ---
    covers_more = set(class_names) != set(COCO_SHARED)
    print("\nCOCO evaluation, all test images:")
    overall = coco_evaluate(gt_dict, predictions, verbose=True)
    overall["mAP_shared"], overall["mAP50_shared"] = shared_means(overall)
    print(f"{'class':15}{'GT boxes':>10}{'AP50':>8}{'AP':>8}")
    for name, r in sorted(overall["per_class"].items(), key=lambda kv: -kv[1]["gt_boxes"]):
        print(f"{name:15}{r['gt_boxes']:10}{r['AP50']:8.3f}{r['AP']:8.3f}")
    print(f"{'mean':15}{'':10}{overall['mAP50']:8.3f}{overall['mAP']:8.3f}")
    if covers_more:
        print(f"{'mean, 8 shared':15}{'':10}{overall['mAP50_shared']:8.3f}{overall['mAP_shared']:8.3f}"
              f"   <- compare this row with the stock COCO model")

    by_time = {}
    for period in ["daytime", "night", "dawn/dusk"]:
        ids = [image_ids[p] for p in image_ids if timeofday[matches[p]] == period]
        if ids:
            r = coco_evaluate(gt_dict, predictions, ids)
            shared, shared50 = shared_means(r)
            by_time[period] = {"images": len(ids), "mAP": r["mAP"], "mAP50": r["mAP50"],
                               "mAP_shared": shared, "mAP50_shared": shared50}
    print("\nBy time of day:")
    for period, r in by_time.items():
        extra = f"   8 shared classes: mAP50 {r['mAP50_shared']:.3f}  mAP {r['mAP_shared']:.3f}" if covers_more else ""
        print(f"  {period:10} n={r['images']:4}  mAP50 {r['mAP50']:.3f}  mAP {r['mAP']:.3f}{extra}")

    with open(args.results, "w") as f:
        json.dump({"model": args.model, "imgsz": args.imgsz, "images": len(image_ids),
                   "classes": class_names, "cannot_output": cannot_output,
                   "labelled_objects": dict(all_labels),
                   "overall": overall, "by_timeofday": by_time}, f, indent=2)
    print(f"\nSaved {args.results}")


if __name__ == "__main__":
    main()
