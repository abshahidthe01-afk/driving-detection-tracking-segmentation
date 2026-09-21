import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from PIL import Image

# 1. Load Model
CHECKPOINT_PATH = "drivable_finetuned.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
except Exception:
    model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, classes=19)

model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0)
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# 2. Pick a real night test image from your held-out test folder
test_img_dir = Path("bdd100k_drivable_subset/test/images")
test_mask_dir = Path("bdd100k_drivable_subset/test/masks")

# Look through test images to find a dark/night scene
selected_img_path = None
for p in sorted(test_img_dir.glob("*.png")):
    sample = cv2.imread(str(p))
    if sample is not None and sample.mean() < 60:  # Dark / night image
        selected_img_path = p
        break

if selected_img_path is None:
    # Default to first test image if mean brightness check fails
    selected_img_path = test_img_dir / "00000.png"

print(f"Testing on unseen test-set image: {selected_img_path.name}")
raw_img = cv2.imread(str(selected_img_path))
orig_h, orig_w = raw_img.shape[:2]

# 3. Model Inference (320 height x 640 width)
rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (640, 320))

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = resized.astype(np.float32) / 255.0
tensor = (tensor - mean) / std
tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

# Resize mask back to full size
pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# 4. Color Overlay: Green = Direct Lane, Blue = Alternative Lanes
overlay = raw_img.copy()
GREEN = np.array([0, 255, 0], dtype=np.float32)
BLUE  = np.array([255, 0, 0], dtype=np.float32)

mask_0 = (pred_full == 0)
mask_1 = (pred_full == 1)

if np.any(mask_0):
    overlay[mask_0] = (0.5 * overlay[mask_0].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
if np.any(mask_1):
    overlay[mask_1] = (0.5 * overlay[mask_1].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)

# 5. Load the real Ground Truth for comparison if it exists
mask_file = test_mask_dir / selected_img_path.name
if mask_file.exists():
    gt_mask = np.array(Image.open(mask_file))
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    
    gt_overlay = raw_img.copy()
    if np.any(gt_mask == 0):
        gt_overlay[gt_mask == 0] = (0.5 * gt_overlay[gt_mask == 0].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
    if np.any(gt_mask == 1):
        gt_overlay[gt_mask == 1] = (0.5 * gt_overlay[gt_mask == 1].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)
    
    cv2.putText(overlay, "Model Prediction (v1)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(gt_overlay, "Ground Truth (Actual Human Label)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    final_view = np.vstack([overlay, gt_overlay])
else:
    final_view = overlay

output_file = "test_set_night_evaluation.png"
cv2.imwrite(output_file, final_view)
print(f"Done! Saved comparison to: {output_file}")