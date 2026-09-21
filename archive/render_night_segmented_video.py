import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# 1. SETTINGS & PATHS
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"
OUTPUT_PATH = "driving_night_rain_segmented_final.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. LOAD MODEL
print("Loading model...")
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

# 3. OPEN INPUT VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file '{VIDEO_PATH}'.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We crop the windshield section: between 25% height and 75% height
crop_top = int(orig_h * 0.25)
crop_bottom = int(orig_h * 0.75)
crop_h = crop_bottom - crop_top

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

print(f"Rendering video to '{OUTPUT_PATH}' ({total_frames} frames)...")

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GREEN = np.array([0, 255, 0], dtype=np.float32)  # Ego Lane (Direct)
BLUE  = np.array([255, 0, 0], dtype=np.float32)  # Side Lanes (Alternative)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Focus on windshield area
    windshield = frame[crop_top:crop_bottom, :]

    # 2. Convert to RGB and resize to model dimensions (320 height x 640 width)
    rgb = cv2.cvtColor(windshield, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 320))

    # 3. Normalize
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. Run model inference
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 5. Scale the mask back up to match the windshield crop size
    pred_crop = cv2.resize(pred, (orig_w, crop_h), interpolation=cv2.INTER_NEAREST)

    # 6. Apply color overlay to the windshield
    segmented_crop = windshield.copy()
    
    mask_direct = (pred_crop == 0)
    if np.any(mask_direct):
        segmented_crop[mask_direct] = (0.5 * segmented_crop[mask_direct].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)

    mask_alt = (pred_crop == 1)
    if np.any(mask_alt):
        segmented_crop[mask_alt] = (0.5 * segmented_crop[mask_alt].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)

    # 7. Paste the segmented windshield back into the full video frame
    frame[crop_top:crop_bottom, :] = segmented_crop

    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()
print(f"\nDone! Successfully created: {OUTPUT_PATH}")