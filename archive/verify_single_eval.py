import os
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 3
model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0),
)
model.load_state_dict(torch.load("drivable_finetuned.pth", map_location=device, weights_only=True))
model = model.eval().to(device)

test_transform = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])

test_img_dir = "bdd100k_drivable_subset/test/images"
test_mask_dir = "bdd100k_drivable_subset/test/masks"
all_filenames = sorted(os.listdir(test_img_dir))

# Check the FIRST 20 test images, not just one -- specifically look for variety
# in ground truth class balance, to test whether the model just always predicts
# class 2 regardless of what's actually there (class collapse) or genuinely
# tracks the ground truth per-image.
print(f"Checking {min(20, len(all_filenames))} test images for class collapse...\n")

for fname in all_filenames[:20]:
    image = np.array(Image.open(os.path.join(test_img_dir, fname)).convert("RGB"))
    mask = np.array(Image.open(os.path.join(test_mask_dir, fname)))

    transformed = test_transform(image=image, mask=mask)
    input_tensor = torch.as_tensor(transformed["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
    target = np.array(transformed["mask"])

    with torch.no_grad():
        output = model(input_tensor)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    gt_class2_pct = (target == 2).sum() / target.size * 100
    pred_class2_pct = (pred == 2).sum() / pred.size * 100

    ious = []
    for cls in range(NUM_CLASSES):
        intersection = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        ious.append(intersection / union if union > 0 else float('nan'))
    miou = np.nanmean(ious)

    print(f"{fname}: GT class2={gt_class2_pct:5.1f}%  Pred class2={pred_class2_pct:5.1f}%  mIoU={miou:.3f}")