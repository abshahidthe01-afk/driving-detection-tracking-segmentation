"""
Evaluate YOLOv8 object detection on the same 614 held-out test images as the
segmentation model, against BDD100K's own box labels.

The detector uses its stock COCO weights (not fine-tuned on BDD100K), so its classes
are mapped to BDD100K's. COCO has no "traffic sign" and no "rider" class, so those two
BDD100K classes cannot be detected at all; they are left out of the mAP and reported
separately as the share of labelled objects the detector cannot find.

The test images are found in the BDD100K dataset by file content (MD5), so the result
does not depend on reproducing the shuffle that made the split.

Scoring uses pycocotools, the standard COCO evaluation: mAP over IoU 0.50:0.95, mAP at
IoU 0.50, per class, and split by time of day.

Needs:  pip install pycocotools
Run:    python evaluate_detection.py [--model yolov8s.pt] [--imgsz 640]
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
RESULTS_PATH = "detection_results.json"

# COCO class (as YOLOv8 names it) -> BDD100K class. "person" gets whatever this dataset
# calls pedestrians ("pedestrian" or "person"), decided at run time.
COCO_TO_BDD = {
    "person": None,
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
}
# BDD100K classes with no COCO counterpart: the stock detector cannot find them.
NOT_IN_COCO = ["traffic sign", "rider"]


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
    args = parser.parse_args()

    import torch
    import fiftyone.utils.huggingface as fouh
    from ultralytics import YOLO

    # --- Find the test images in BDD100K by content ---
    test_files = sorted(os.listdir(TEST_IMG_DIR))
    test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]
    print(f"{len(test_paths)} test images in {TEST_IMG_DIR}")

    print("Loading BDD100K (local Hugging Face cache)...")
    dataset = fouh.load_from_hub("dgural/bdd100k")
    filepaths = dataset.values("filepath")
    labels = dataset.values("detections.detections.label")
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
        all_labels.update(labels[i] or [])
    person_label = "pedestrian" if all_labels["pedestrian"] >= all_labels["person"] else "person"
    coco_to_bdd = dict(COCO_TO_BDD, person=person_label)
    class_names = sorted(set(coco_to_bdd.values()))

    print("\nLabelled objects in the test images:")
    for name, count in all_labels.most_common():
        note = "" if name in class_names else ("  <- no COCO class" if name in NOT_IN_COCO else "  <- not evaluated")
        print(f"  {name:15} {count:6}{note}")
    total = sum(all_labels.values())
    missing = sum(all_labels[n] for n in NOT_IN_COCO)
    print(f"  {missing}/{total} labelled objects ({100 * missing / max(total, 1):.1f}%) are classes "
          f"the COCO-trained detector cannot output ({', '.join(NOT_IN_COCO)})")

    images = []
    image_ids = {}
    for image_id, (fname, path) in enumerate(zip(test_files, test_paths), start=1):
        if path not in matches:
            continue
        i = matches[path]
        width, height = Image.open(path).size
        images.append({"id": image_id, "file_name": fname, "width": width, "height": height,
                       "boxes": list(zip(labels[i] or [], boxes[i] or []))})
        image_ids[path] = image_id
    gt_dict = build_coco_gt(images, class_names)
    cat_id = {c["name"]: c["id"] for c in gt_dict["categories"]}

    # --- Predictions ---
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nRunning {args.model} at imgsz={args.imgsz} on {'GPU' if device == 0 else 'CPU'}...")
    model = YOLO(args.model)
    predictions = []
    for n, (path, image_id) in enumerate(image_ids.items(), start=1):
        # conf=0.001 and iou=0.7 are Ultralytics' own settings for computing mAP
        result = model.predict(path, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                               device=device, verbose=False)[0]
        for (x1, y1, x2, y2), score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(),
                                                result.boxes.cls.cpu().numpy()):
            bdd = coco_to_bdd.get(model.names[int(cls)])
            if bdd is None:
                continue
            predictions.append({"image_id": image_id, "category_id": cat_id[bdd],
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(score)})
        if n % 100 == 0:
            print(f"  {n}/{len(image_ids)}")

    # --- Scores ---
    print("\nCOCO evaluation, all test images:")
    overall = coco_evaluate(gt_dict, predictions, verbose=True)
    print(f"{'class':15}{'GT boxes':>10}{'AP50':>8}{'AP':>8}")
    for name, r in sorted(overall["per_class"].items(), key=lambda kv: -kv[1]["gt_boxes"]):
        print(f"{name:15}{r['gt_boxes']:10}{r['AP50']:8.3f}{r['AP']:8.3f}")
    print(f"{'mean':15}{'':10}{overall['mAP50']:8.3f}{overall['mAP']:8.3f}")

    by_time = {}
    for period in ["daytime", "night", "dawn/dusk"]:
        ids = [image_ids[p] for p in image_ids if timeofday[matches[p]] == period]
        if ids:
            r = coco_evaluate(gt_dict, predictions, ids)
            by_time[period] = {"images": len(ids), "mAP": r["mAP"], "mAP50": r["mAP50"]}
    print("\nBy time of day:")
    for period, r in by_time.items():
        print(f"  {period:10} n={r['images']:4}  mAP50 {r['mAP50']:.3f}  mAP {r['mAP']:.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"model": args.model, "imgsz": args.imgsz, "images": len(image_ids),
                   "labelled_objects": dict(all_labels), "not_in_coco": NOT_IN_COCO,
                   "overall": overall, "by_timeofday": by_time}, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
