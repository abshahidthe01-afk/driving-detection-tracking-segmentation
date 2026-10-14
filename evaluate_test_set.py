import os
import json
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

DATA_ROOT = "bdd100k_drivable_subset"
NUM_CLASSES = 3
INPUT_HEIGHT, INPUT_WIDTH = 320, 640

# BDD100K drivable-area mask values, verified with definitive_class_check.py
# (polygon-to-pixel cross-check) and used the same way in pipeline.py.
# The order matters: every per-class number below is reported in this order.
DIRECT, ALTERNATIVE, BACKGROUND = 0, 1, 2
CLASS_NAMES = ["direct (ego lane)", "alternative (adjacent lane)", "background"]

# Cityscapes train ID of "road" in the 19-class baseline model
CITYSCAPES_ROAD = 0

test_transform = A.Compose([
    A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])


def load_finetuned_model():
    model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
    model.segmentation_head = nn.Sequential(
        nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
        nn.UpsamplingBilinear2d(scale_factor=4.0),
    )
    model.load_state_dict(torch.load("drivable_finetuned.pth", map_location=device, weights_only=True))
    return model.eval().to(device)


def load_baseline_model():
    # Original 19-class Cityscapes model, no fine-tuning. Its "road" class is mapped to
    # DIRECT and everything else to BACKGROUND (see evaluate()). Cityscapes has no notion
    # of ego vs adjacent lane, so the 3-class numbers are only an approximation for it;
    # the binary drivable IoU (direct + alternative vs background) is the fairer comparison.
    model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
    return model.eval().to(device)


def compute_iou_per_class(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            ious.append(np.nan)  # class not present in this frame -- exclude from average
        else:
            ious.append(intersection / union)
    return ious


def dataset_iou_per_class(confusion):
    """Standard (dataset-level) IoU: intersections and unions summed over all test images."""
    intersection = np.diag(confusion).astype(float)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - np.diag(confusion)
    with np.errstate(invalid="ignore", divide="ignore"):
        return intersection / union


def drivable_iou(confusion):
    """Binary IoU of drivable (direct + alternative) against background, over all test images."""
    drivable = [DIRECT, ALTERNATIVE]
    intersection = confusion[np.ix_(drivable, drivable)].sum()
    union = confusion[drivable, :].sum() + confusion[:, drivable].sum() - intersection
    return intersection / union


def evaluate(model, is_baseline, test_img_dir, test_mask_dir):
    filenames = sorted(os.listdir(test_img_dir))
    all_ious = []
    # rows: true class, columns: predicted class, summed over all test images
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for i, fname in enumerate(filenames):
            image = np.array(Image.open(os.path.join(test_img_dir, fname)).convert("RGB"))
            mask = np.array(Image.open(os.path.join(test_mask_dir, fname)))

            transformed = test_transform(image=image, mask=mask)
            input_tensor = torch.as_tensor(transformed["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
            target = transformed["mask"]

            output = model(input_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

            if is_baseline:
                # Cityscapes "road" -> DIRECT, everything else -> BACKGROUND. The baseline
                # has no equivalent for "alternative", so it can never predict that class.
                mapped_pred = np.full_like(pred, BACKGROUND)
                mapped_pred[pred == CITYSCAPES_ROAD] = DIRECT
                pred = mapped_pred

            ious = compute_iou_per_class(pred, target, NUM_CLASSES)
            all_ious.append(ious)

            valid = target < NUM_CLASSES
            confusion += np.bincount(
                NUM_CLASSES * target[valid].astype(np.int64) + pred[valid].astype(np.int64),
                minlength=NUM_CLASSES ** 2,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(filenames)}")

    all_ious = np.array(all_ious, dtype=float)
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    overall_miou = np.nanmean(mean_iou_per_class)
    return mean_iou_per_class, overall_miou, all_ious, filenames, confusion


def breakdown_by_condition(all_ious, filenames, metadata_by_filename, label):
    """Group per-sample mIoU by weather and by timeofday, print averages for each group."""
    per_sample_miou = np.nanmean(all_ious, axis=1)  # average across classes, per sample

    groups = {}
    for fname, miou in zip(filenames, per_sample_miou):
        meta = metadata_by_filename.get(fname, {})
        for key in ["weather", "timeofday"]:
            value = meta.get(key, "unknown")
            groups.setdefault((key, value), []).append(miou)

    print(f"  --- {label}: breakdown by condition (per-image mIoU) ---")
    for (key, value), scores in sorted(groups.items()):
        print(f"  {key}={value}: mean IoU = {np.nanmean(scores):.4f} (n={len(scores)})")


def report(label, per_class, miou, all_ious, filenames, confusion):
    """Print both metrics for one model and return its dataset-level numbers for the summary."""
    print(f"  Per-image IoU (averaged over images, classes absent from an image skipped):")
    for name, iou in zip(CLASS_NAMES, per_class):
        print(f"    {name}: {iou:.4f}")
    print(f"    Mean IoU: {miou:.4f}")

    dataset_per_class = dataset_iou_per_class(confusion)
    dataset_miou = np.nanmean(dataset_per_class)
    binary = drivable_iou(confusion)
    print(f"  Dataset-level IoU (standard: summed over all test images):")
    for name, iou in zip(CLASS_NAMES, dataset_per_class):
        print(f"    {name}: {iou:.4f}")
    print(f"    Mean IoU: {dataset_miou:.4f}")
    print(f"  Drivable vs background IoU (binary, dataset-level): {binary:.4f}")
    breakdown_by_condition(all_ious, filenames, metadata_by_filename, label)
    return miou, dataset_miou, binary


test_img_dir = os.path.join(DATA_ROOT, "test", "images")
test_mask_dir = os.path.join(DATA_ROOT, "test", "masks")
print(f"Evaluating on {len(os.listdir(test_img_dir))} held-out test images (never used in training or validation)\n")

with open(os.path.join(DATA_ROOT, "test", "metadata.json")) as f:
    metadata_list = json.load(f)
metadata_by_filename = {m["filename"]: m for m in metadata_list}

print("=== Baseline (original Cityscapes-pretrained, no fine-tuning) ===")
baseline = report("Baseline", *evaluate(
    load_baseline_model(), is_baseline=True, test_img_dir=test_img_dir, test_mask_dir=test_mask_dir))
print()

print("=== Fine-tuned model ===")
finetuned = report("Fine-tuned", *evaluate(
    load_finetuned_model(), is_baseline=False, test_img_dir=test_img_dir, test_mask_dir=test_mask_dir))
print()

print("=== Summary ===")
print(f"{'':28}{'Baseline':>10}{'Fine-tuned':>12}")
for name, b, f in zip(["Per-image mIoU", "Dataset-level mIoU", "Drivable IoU (binary)"], baseline, finetuned):
    print(f"{name:28}{b:10.4f}{f:12.4f}")
