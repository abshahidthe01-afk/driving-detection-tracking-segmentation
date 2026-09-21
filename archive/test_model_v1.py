import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# 1. Update this path to your saved v1 checkpoint file (.pth or .pt)
CHECKPOINT_PATH = "best_model_v1.pth" 

# Test image: 00001 from your dataset
IMAGE_PATH = "bdd100k_drivable_subset/train/images/00001.png"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate SegFormer model architecture with 3 classes
import segmentation_models_pytorch as smp
import torch.nn as nn

print("Building model...")
model = smp.Segformer(
    encoder_name="mit_b0",
    encoder_weights=None,
    classes=3
)
# Ensure head matches your architecture
model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0)
)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# 2. Load and preprocess image
raw_img = cv2.imread(IMAGE_PATH)
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb_img, (640, 320))

# ImageNet normalization
norm_tensor = resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm_tensor = (norm_tensor - mean) / std

# [1, 3, H, W]
tensor = torch.tensor(norm_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

# 3. Predict
with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Resize prediction mask back to original image size
pred = cv2.resize(pred.astype(np.uint8), (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)

# 4. Color the prediction using the CORRECT mapping
# Class 0: Green (Own Lane)
# Class 1: Blue (Other Lanes)
# Class 2: Transparent (Background)
overlay = raw_img.copy()
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]

overlay[pred == 0] = cv2.addWeighted(overlay[pred == 0], 0.5, np.full_like(overlay[pred == 0], GREEN), 0.5, 0)
overlay[pred == 1] = cv2.addWeighted(overlay[pred == 1], 0.5, np.full_like(overlay[pred == 1], BLUE), 0.5, 0)

output_name = "v1_prediction_result.png"
cv2.imwrite(output_name, overlay)
print(f"Done! Saved prediction overlay to: {output_name}")