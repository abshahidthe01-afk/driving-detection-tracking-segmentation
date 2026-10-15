"""
Make a video of the detector's results on the 614 test images, to look at them by eye.

Each frame is one test image:
  green   labelled object the detector found (same class, box overlap IoU >= 0.5),
          labelled with the class and confidence of the detection that found it
  red     labelled object it missed
  orange  labelled object of a class the model cannot output (traffic sign and rider
          for the stock COCO model; none after fine-tuning on BDD100K)
  blue    detection with no matching label (false alarm)

Detections are shown above the pipeline's confidence threshold (0.25), so the video
matches what pipeline.py draws. (The mAP in evaluate_detection.py uses all confidences.)

Run:    python visualize_detection.py [--model yolov8s.pt] [--only night] [--worst] [--limit 100]
                                      [--fps 1] [--output detection_review.mp4]
"""

import argparse
import os

import cv2
import numpy as np

from evaluate_detection import BDD_CLASSES, TEST_IMG_DIR, bdd_mapping, match_by_content, normalize_label

CONF = 0.25          # Ultralytics' default, as used by pipeline.py
IOU_MATCH = 0.5
BANNER, LEGEND = 64, 40   # pixel heights of the text bars above and below the image

# BGR colours
GREEN, RED, ORANGE, BLUE = (60, 200, 60), (40, 40, 230), (0, 165, 255), (230, 140, 30)
WHITE, DARK = (255, 255, 255), (32, 32, 32)


def iou(a, b):
    """IoU of two boxes given as (x1, y1, x2, y2)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, preds, threshold=IOU_MATCH):
    """
    Greedy matching, highest confidence first, as in COCO evaluation.

    gt: list of (label, box); preds: list of (label, box, confidence).
    Returns (found_by, matched): for each label the index of the detection that found it
    (None if missed), and for each detection whether it matched a label. Labels and
    detections match only within a class.
    """
    found_by = [None] * len(gt)
    matched = [False] * len(preds)
    for p in sorted(range(len(preds)), key=lambda k: -preds[k][2]):
        label, box, _ = preds[p]
        best, best_iou = None, threshold
        for g, (g_label, g_box) in enumerate(gt):
            if found_by[g] is not None or g_label != label:
                continue
            overlap = iou(box, g_box)
            if overlap >= best_iou:
                best, best_iou = g, overlap
        if best is not None:
            found_by[best] = p
            matched[p] = True
    return found_by, matched


def text(img, s, org, colour=WHITE, scale=0.55, thickness=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def box(img, b, colour, label=None, thickness=2, below=False):
    """Draw a box; its label goes above it, or below it with below=True."""
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        if below:
            y = min(y2 + th + 4, img.shape[0])
        else:
            y = max(y1, th + 4)
        cv2.rectangle(img, (x1, y - th - 4), (x1 + tw + 4, y), colour, -1)
        text(img, label, (x1 + 2, y - 3), DARK, 0.45)


def draw_frame(image, gt, uncoverable, preds, found_by, matched, title):
    """
    Return the image with a title bar, the boxes and a legend.

    gt: (label, box) the detector could find; uncoverable: (label, box) of classes it
    cannot output; preds: (label, box, confidence); found_by / matched from match().
    """
    h, w = image.shape[:2]
    frame = np.full((BANNER + h + LEGEND, w, 3), DARK, dtype=np.uint8)
    canvas = image.copy()
    for label, b in uncoverable:
        box(canvas, b, ORANGE, None, 1)
    for (label, b, conf), ok in zip(preds, matched):
        if not ok:
            box(canvas, b, BLUE, f"{label} {conf:.2f}", 2)
    # drawn after the false alarms, so found objects' labels stay readable; labels of missed
    # objects go below their box, the others above
    for (label, b), p in zip(gt, found_by):
        if p is None:
            box(canvas, b, RED, label, 2, below=True)
        else:
            box(canvas, b, GREEN, f"{label} {preds[p][2]:.2f}", 2)
    frame[BANNER:BANNER + h] = canvas

    n_found = sum(p is not None for p in found_by)
    text(frame, title, (12, 24), WHITE, 0.6)
    stats = (f"labelled {len(gt) + len(uncoverable)}   found {n_found}   missed {len(gt) - n_found}   "
             f"class not in model {len(uncoverable)}   false alarms {matched.count(False)}")
    text(frame, stats, (12, 52), WHITE, 0.55)

    x, y = 12, BANNER + h + 26
    for colour, name in [(GREEN, "found"), (RED, "missed"), (ORANGE, "class the model cannot output"),
                         (BLUE, "false alarm")]:
        cv2.rectangle(frame, (x, y - 12), (x + 18, y + 2), colour, -1)
        text(frame, name, (x + 26, y), WHITE, 0.5)
        x += 40 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--only", choices=["daytime", "night", "dawn/dusk"], help="only this time of day")
    parser.add_argument("--worst", action="store_true", help="order frames by most missed objects first")
    parser.add_argument("--limit", type=int, help="at most this many frames")
    parser.add_argument("--fps", type=float, default=1.0, help="frames per second (default 1: one image per second)")
    parser.add_argument("--output", default="detection_review.mp4")
    args = parser.parse_args()

    import torch
    import fiftyone.utils.huggingface as fouh
    from ultralytics import YOLO

    test_files = sorted(os.listdir(TEST_IMG_DIR))
    test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]
    print("Loading BDD100K (local Hugging Face cache)...")
    dataset = fouh.load_from_hub("dgural/bdd100k")
    filepaths = dataset.values("filepath")
    labels = [[normalize_label(x) for x in (ls or [])] for ls in dataset.values("detections.detections.label")]
    boxes = dataset.values("detections.detections.bounding_box")
    timeofday = dataset.values("timeofday.label")
    weather = dataset.values("weather.label")
    matches = match_by_content(test_paths, filepaths)
    print(f"Matched {len(matches)}/{len(test_paths)} test images")

    selected = [(f, p) for f, p in zip(test_files, test_paths)
                if p in matches and (args.only is None or timeofday[matches[p]] == args.only)]
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    mapping = bdd_mapping(model.names)
    detectable = set(mapping.values())

    # Pass 1: detections and matching for every selected image (small, kept in memory)
    items = []
    print(f"Running {args.model} at imgsz={args.imgsz} on {len(selected)} images...")
    for n, (fname, path) in enumerate(selected, start=1):
        i = matches[path]
        h, w = cv2.imread(path).shape[:2]
        gt, uncoverable = [], []
        for label, (x, y, bw, bh) in zip(labels[i], boxes[i] or []):
            b = (x * w, y * h, (x + bw) * w, (y + bh) * h)
            if label in detectable:
                gt.append((label, b))
            elif label in BDD_CLASSES:
                uncoverable.append((label, b))
        result = model.predict(path, imgsz=args.imgsz, conf=CONF, device=device, verbose=False)[0]
        preds = []
        for b, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy()):
            label = mapping.get(int(cls))
            if label is not None:
                preds.append((label, tuple(float(v) for v in b), float(conf)))
        found_by, matched = match(gt, preds)
        items.append((fname, path, gt, uncoverable, preds, found_by, matched, timeofday[i], weather[i]))
        if n % 100 == 0:
            print(f"  {n}/{len(selected)}")

    if args.worst:
        items.sort(key=lambda it: -sum(p is None for p in it[5]))
    if args.limit:
        items = items[:args.limit]

    # Pass 2: draw and write the video
    writer = None
    for fname, path, gt, uncoverable, preds, found_by, matched, tod, wea in items:
        frame = draw_frame(cv2.imread(path), gt, uncoverable, preds, found_by, matched,
                           f"{fname}   {tod}, {wea}")
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    if writer is not None:
        writer.release()

    total = sum(len(it[2]) for it in items)
    found_total = sum(sum(p is not None for p in it[5]) for it in items)
    print(f"\nAt confidence {CONF}: found {found_total}/{total} labelled objects of classes it can detect "
          f"({100 * found_total / max(total, 1):.1f}%), "
          f"{sum(len(it[3]) for it in items)} of classes it cannot output, "
          f"{sum(it[6].count(False) for it in items)} false alarms")
    print(f"Wrote {args.output}: {len(items)} frames at {args.fps:g} per second")


if __name__ == "__main__":
    main()
