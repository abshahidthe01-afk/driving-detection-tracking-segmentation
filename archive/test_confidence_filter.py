import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# 1. Load Model
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"

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

# 2. Grab frame 60
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
cap.release()

orig_h, orig_w = frame.shape[:2]

# Windshield crop (excluding ceiling and dashboard)
crop_top = int(orig_h * 0.15)
crop_bottom = int(orig_h * 0.55)
crop_h = crop_bottom - crop_top
windshield = frame[crop_top:crop_bottom, :]

# 3. Model Prediction
rgb = cv2.cvtColor(windshield, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (640, 320))

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = resized.astype(np.float32) / 255.0
tensor = (tensor - mean) / std
tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)
    # Convert logits into real probabilities (0.0 to 1.0)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

# Resize probabilities to match crop size
prob_ego  = cv2.resize(probs[0], (orig_w, crop_h), interpolation=cv2.INTER_LINEAR)
prob_alt  = cv2.resize(probs[1], (orig_w, crop_h), interpolation=cv2.INTER_LINEAR)
prob_bg   = cv2.resize(probs[2], (orig_w, crop_h), interpolation=cv2.INTER_LINEAR)

# 4. Generate comparison images with 3 different confidence thresholds
# 50% sure, 65% sure, and 80% sure
thresholds = [0.50, 0.65, 0.80]
GREEN = np.array([0, 255, 0], dtype=np.float32)
BLUE  = np.array([255, 0, 0], dtype=np.float32)

for thresh in thresholds:
    segmented_windshield = windshield.copy()
    
    # Only paint if probability is strictly GREATER than the threshold
    ego_mask = (prob_ego > thresh) & (prob_ego > prob_alt) & (prob_ego > prob_bg)
    alt_mask = (prob_alt > thresh) & (prob_alt > prob_ego) & (prob_alt > prob_bg)
    
    if np.any(ego_mask):
        segmented_windshield[ego_mask] = (0.5 * segmented_windshield[ego_mask].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
    if np.any(alt_mask):
        segmented_windshield[alt_mask] = (0.5 * segmented_windshield[alt_mask].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)
        
    result_frame = frame.copy()
    result_frame[crop_top:crop_bottom, :] = segmented_windshield
    
    # Label the image
    cv2.putText(result_frame, f"Confidence Threshold: {int(thresh * 100)}%", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    out_name = f"threshold_{int(thresh * 100)}.png"
    cv2.imwrite(out_name, result_frame)
    print(f"Saved: {out_name}")

print("\nAll 3 threshold images created. Open them to compare!")