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
CLASS_NAMES = ["background", "direct", "alternative"]

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
    model.load_state_dict(torch.load("drivable_finetuned.pth", map_location=device))
    return model.eval().to(device)


def load_baseline_model():
    # Original 19-class Cityscapes model. We map its "road" class (0) to our
    # "direct" class (1) for a fair-ish comparison -- it was never trained on
    # our 3-class scheme, so this is an approximation, not a perfect apples-to-apples test.
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


def evaluate(model, is_baseline, test_img_dir, test_mask_dir):
    filenames = sorted(os.listdir(test_img_dir))
    all_ious = []

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
                # Baseline outputs 19 Cityscapes classes. Map class 0 ("road") to our
                # "direct" (1); everything else becomes "background" (0). We have no
                # equivalent for "alternative" in the baseline, so it can never predict that class.
                mapped_pred = np.zeros_like(pred)
                mapped_pred[pred == 0] = 1  # Cityscapes "road" -> our "direct"
                pred = mapped_pred

            ious = compute_iou_per_class(pred, target, NUM_CLASSES)
            all_ious.append(ious)

            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(filenames)}")

    all_ious = np.array(all_ious, dtype=float)
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    overall_miou = np.nanmean(mean_iou_per_class)
    return mean_iou_per_class, overall_miou, all_ious, filenames


def breakdown_by_condition(all_ious, filenames, metadata_by_filename, label):
    """Group per-sample mIoU by weather and by timeofday, print averages for each group."""
    per_sample_miou = np.nanmean(all_ious, axis=1)  # average across classes, per sample

    groups = {}
    for fname, miou in zip(filenames, per_sample_miou):
        meta = metadata_by_filename.get(fname, {})
        for key in ["weather", "timeofday"]:
            value = meta.get(key, "unknown")
            groups.setdefault((key, value), []).append(miou)

    print(f"  --- {label}: breakdown by condition ---")
    for (key, value), scores in sorted(groups.items()):
        print(f"  {key}={value}: mean IoU = {np.nanmean(scores):.4f} (n={len(scores)})")


test_img_dir = os.path.join(DATA_ROOT, "test", "images")
test_mask_dir = os.path.join(DATA_ROOT, "test", "masks")
print(f"Evaluating on {len(os.listdir(test_img_dir))} held-out test images (never used in training or validation)\n")

with open(os.path.join(DATA_ROOT, "test", "metadata.json")) as f:
    metadata_list = json.load(f)
metadata_by_filename = {m["filename"]: m for m in metadata_list}

print("=== Baseline (original Cityscapes-pretrained, no fine-tuning) ===")
baseline_model = load_baseline_model()
baseline_per_class, baseline_miou, baseline_all_ious, baseline_filenames = evaluate(
    baseline_model, is_baseline=True, test_img_dir=test_img_dir, test_mask_dir=test_mask_dir)
for name, iou in zip(CLASS_NAMES, baseline_per_class):
    print(f"  {name}: {iou:.4f}")
print(f"  Mean IoU: {baseline_miou:.4f}")
breakdown_by_condition(baseline_all_ious, baseline_filenames, metadata_by_filename, "Baseline")
print()

print("=== Fine-tuned model ===")
finetuned_model = load_finetuned_model()
finetuned_per_class, finetuned_miou, finetuned_all_ious, finetuned_filenames = evaluate(
    finetuned_model, is_baseline=False, test_img_dir=test_img_dir, test_mask_dir=test_mask_dir)
for name, iou in zip(CLASS_NAMES, finetuned_per_class):
    print(f"  {name}: {iou:.4f}")
print(f"  Mean IoU: {finetuned_miou:.4f}")
breakdown_by_condition(finetuned_all_ious, finetuned_filenames, metadata_by_filename, "Fine-tuned")
print()

print("=== Summary ===")
print(f"Baseline mIoU:   {baseline_miou:.4f}")
print(f"Fine-tuned mIoU: {finetuned_miou:.4f}")
print(f"Absolute improvement: {finetuned_miou - baseline_miou:+.4f}")
