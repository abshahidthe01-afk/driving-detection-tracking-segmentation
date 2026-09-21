import urllib.request
import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

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

# 2. Download standard landscape night dashcam image with browser header
image_url = "https://images.pexels.com/photos/1542493/pexels-photo-1542493.jpeg?auto=compress&cs=tinysrgb&w=1280"
dashcam_path = "sample_night_dashcam.jpg"

print("Downloading standard landscape night driving image...")
req = urllib.request.Request(
    image_url,
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)
with urllib.request.urlopen(req) as response, open(dashcam_path, "wb") as out_file:
    out_file.write(response.read())

raw_img = cv2.imread(dashcam_path)
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

# Resize mask back to original resolution
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

output_file = "night_dashcam_result.png"
cv2.imwrite(output_file, overlay)
print(f"\nDone! Saved output to '{output_file}'. Open it to see the result.")