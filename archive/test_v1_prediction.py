import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path

# Paths
CHECKPOINT_PATH = "drivable_finetuned.pth"
IMAGE_PATH = "bdd100k_drivable_subset/train/images/00001.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Recreate the model architecture
print("Loading model architecture...")
try:
    model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
except Exception:
    model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, classes=19)

# Replace the head with the 3-class head you trained
model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0)
)

# 2. Load the v1 weights
print(f"Loading weights from: {CHECKPOINT_PATH}")
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

# 3. Load and prepare the test image
raw_img = cv2.imread(IMAGE_PATH)
if raw_img is None:
    print(f"Could not open image at {IMAGE_PATH}")
    exit()

# Resize to your model's input size (320 height, 640 width)
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb_img, (640, 320))

# Normalize with standard ImageNet values
tensor = resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = (tensor - mean) / std

# Convert to PyTorch shape: [1, Channels, Height, Width]
tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

# 4. Predict
print("Running prediction...")
with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Resize prediction mask back to original picture dimensions
orig_h, orig_w = raw_img.shape[:2]
pred = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# 5. Draw the overlay with the CORRECT mapping:
# Class 0 = Ego/Direct Lane (Green)
# Class 1 = Other/Alternative Lanes (Blue)
# Class 2 = Background/Sky/Buildings (Natural / Transparent)
overlay = raw_img.copy()
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]

overlay[pred == 0] = cv2.addWeighted(overlay[pred == 0], 0.5, np.full_like(overlay[pred == 0], GREEN), 0.5, 0)
overlay[pred == 1] = cv2.addWeighted(overlay[pred == 1], 0.5, np.full_like(overlay[pred == 1], BLUE), 0.5, 0)

output_file = "v1_prediction_result.png"
cv2.imwrite(output_file, overlay)
print(f"\nSuccess! Saved prediction to: {output_file}")