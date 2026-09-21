import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# 1. Load Model
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 2. Grab frame 60 (2 seconds into video)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video.")
    exit()

orig_h, orig_w = frame.shape[:2]

# 3. Focus on the Windshield (shift crop higher up, excluding the dashboard)
crop_top = int(orig_h * 0.15)     # Cut dark car roof
crop_bottom = int(orig_h * 0.55)  # Cut speedometer and steering wheel
crop_h = crop_bottom - crop_top

windshield = frame[crop_top:crop_bottom, :]

# 4. Prepare for model (320 height x 640 width)
rgb = cv2.cvtColor(windshield, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (640, 320))

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = resized.astype(np.float32) / 255.0
tensor = (tensor - mean) / std
tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

# 5. Predict
with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

# Resize mask back to windshield crop dimensions
pred_crop = cv2.resize(pred, (orig_w, crop_h), interpolation=cv2.INTER_NEAREST)

# 6. Apply Colors: Green = Own Lane, Blue = Side Lane
GREEN = np.array([0, 255, 0], dtype=np.float32)
BLUE  = np.array([255, 0, 0], dtype=np.float32)

segmented_windshield = windshield.copy()

mask_ego = (pred_crop == 0)
if np.any(mask_ego):
    segmented_windshield[mask_ego] = (0.5 * segmented_windshield[mask_ego].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)

mask_alt = (pred_crop == 1)
if np.any(mask_alt):
    segmented_windshield[mask_alt] = (0.5 * segmented_windshield[mask_alt].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)

# 7. Paste back into full frame
result_frame = frame.copy()
result_frame[crop_top:crop_bottom, :] = segmented_windshield

# Draw a red line showing where the crop cut off the dashboard
cv2.line(result_frame, (0, crop_bottom), (orig_w, crop_bottom), (0, 0, 255), 2)
cv2.putText(result_frame, "Dashboard cut-off line", (20, crop_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

output_name = "test_dashboard_fixed.png"
cv2.imwrite(output_name, result_frame)
print(f"\nDone! Open '{output_name}' to check the result.")