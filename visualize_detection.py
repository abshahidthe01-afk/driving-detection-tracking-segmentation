"""
Make a video of the detector's results on the 614 test images, to look at them by eye.

Each frame is one test image:
  green   labelled object the detector found (same class, box overlap IoU >= 0.5)
  red     labelled object it missed
  orange  labelled traffic sign or rider: classes the COCO-trained detector cannot output
  blue    detection with no matching label (false alarm)

Detections are shown above the pipeline's confidence threshold (0.25), so the video
matches what pipeline.py draws. (The mAP in evaluate_detection.py uses all confidences.)

Run:    python visualize_detection.py [--only night] [--worst] [--limit 100] [--fps 1]
Writes: detection_review.mp4
"""

import argparse
import os

import cv2
import numpy as np

from evaluate_detection import COCO_TO_BDD, NOT_IN_COCO, TEST_IMG_DIR, match_by_content

OUTPUT_PATH = "detection_review.mp4"
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
    Returns (found, matched): one flag per label (found by some detection) and one per
    detection (matched to a label). Labels and detections match only within a class.
    """
    found = [False] * len(gt)
    matched = [False] * len(preds)
    for p in sorted(range(len(preds)), key=lambda k: -preds[k][2]):
        label, box, _ = preds[p]
        best, best_iou = None, threshold
        for g, (g_label, g_box) in enumerate(gt):
            if found[g] or g_label != label:
                continue
            overlap = iou(box, g_box)
            if overlap >= best_iou:
                best, best_iou = g, overlap
        if best is not None:
            found[best] = matched[p] = True
    return found, matched


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


def draw_frame(image, gt, uncoverable, preds, found, matched, title):
    """
    Return the image with a title bar, the boxes and a legend.

    gt: (label, box) the detector could find; uncoverable: (label, box) of classes it
    cannot output; preds: (label, box, confidence); found / matched from match().
    """
    h, w = image.shape[:2]
    frame = np.full((BANNER + h + LEGEND, w, 3), DARK, dtype=np.uint8)
    canvas = image.copy()
    for label, b in uncoverable:
        box(canvas, b, ORANGE, None, 1)
    for (label, b), ok in zip(gt, found):
        # missed labels go below the box, detection labels above: they never cover each other
        box(canvas, b, GREEN if ok else RED, None if ok else label, 2, below=True)
    for (label, b, conf), ok in zip(preds, matched):
        if not ok:
            box(canvas, b, BLUE, f"{label} {conf:.2f}", 2)
    frame[BANNER:BANNER + h] = canvas

    n_found = sum(found)
    text(frame, title, (12, 24), WHITE, 0.6)
    stats = (f"labelled {len(gt) + len(uncoverable)}   found {n_found}   missed {len(gt) - n_found}   "
             f"no COCO class {len(uncoverable)}   false alarms {matched.count(False)}")
    text(frame, stats, (12, 52), WHITE, 0.55)

    x, y = 12, BANNER + h + 26
    for colour, name in [(GREEN, "found"), (RED, "missed"), (ORANGE, "sign / rider (no COCO class)"),
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
    args = parser.parse_args()

    import torch
    import fiftyone.utils.huggingface as fouh
    from ultralytics import YOLO

    test_files = sorted(os.listdir(TEST_IMG_DIR))
    test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]
    print("Loading BDD100K (local Hugging Face cache)...")
    dataset = fouh.load_from_hub("dgural/bdd100k")
    filepaths = dataset.values("filepath")
    labels = dataset.values("detections.detections.label")
    boxes = dataset.values("detections.detections.bounding_box")
    timeofday = dataset.values("timeofday.label")
    weather = dataset.values("weather.label")
    matches = match_by_content(test_paths, filepaths)
    print(f"Matched {len(matches)}/{len(test_paths)} test images")

    person = "pedestrian" if any("pedestrian" in (labels[i] or []) for i in matches.values()) else "person"
    coco_to_bdd = dict(COCO_TO_BDD, person=person)
    detectable = set(coco_to_bdd.values())

    selected = [(f, p) for f, p in zip(test_files, test_paths)
                if p in matches and (args.only is None or timeofday[matches[p]] == args.only)]
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)

    # Pass 1: detections and matching for every selected image (small, kept in memory)
    items = []
    print(f"Running {args.model} at imgsz={args.imgsz} on {len(selected)} images...")
    for n, (fname, path) in enumerate(selected, start=1):
        i = matches[path]
        h, w = cv2.imread(path).shape[:2]
        gt, uncoverable = [], []
        for label, (x, y, bw, bh) in zip(labels[i] or [], boxes[i] or []):
            b = (x * w, y * h, (x + bw) * w, (y + bh) * h)
            if label in detectable:
                gt.append((label, b))
            elif label in NOT_IN_COCO:
                uncoverable.append((label, b))
        result = model.predict(path, imgsz=args.imgsz, conf=CONF, device=device, verbose=False)[0]
        preds = []
        for b, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                result.boxes.cls.cpu().numpy()):
            label = coco_to_bdd.get(model.names[int(cls)])
            if label is not None:
                preds.append((label, tuple(float(v) for v in b), float(conf)))
        found, matched = match(gt, preds)
        items.append((fname, path, gt, uncoverable, preds, found, matched, timeofday[i], weather[i]))
        if n % 100 == 0:
            print(f"  {n}/{len(selected)}")

    if args.worst:
        items.sort(key=lambda it: -(len(it[2]) - sum(it[5])))
    if args.limit:
        items = items[:args.limit]

    # Pass 2: draw and write the video
    writer = None
    for fname, path, gt, uncoverable, preds, found, matched, tod, wea in items:
        frame = draw_frame(cv2.imread(path), gt, uncoverable, preds, found, matched,
                           f"{fname}   {tod}, {wea}")
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    if writer is not None:
        writer.release()

    total = sum(len(it[2]) for it in items)
    found_total = sum(sum(it[5]) for it in items)
    print(f"\nAt confidence {CONF}: found {found_total}/{total} labelled objects of classes it can detect "
          f"({100 * found_total / max(total, 1):.1f}%), "
          f"{sum(len(it[3]) for it in items)} signs/riders it cannot detect, "
          f"{sum(it[6].count(False) for it in items)} false alarms")
    print(f"Wrote {OUTPUT_PATH}: {len(items)} frames at {args.fps:g} per second")


if __name__ == "__main__":
    main()
