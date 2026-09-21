import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path

# --- 1. SETTINGS ---
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"
OUTPUT_PATH = "night_video_segmented.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. LOAD MODEL ---
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

# --- 3. LETTERBOX HELPER (KEEPS NATURAL PROPORTIONS) ---
def letterbox(image, target_size=(320, 640)):
    target_h, target_w = target_size
    h, w = image.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, pad_top, pad_left, new_h, new_w

# --- 4. PROCESS VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at '{VIDEO_PATH}'.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

print(f"Processing '{VIDEO_PATH}' ({total_frames} frames)...")

frame_idx = 0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GREEN = np.array([0, 255, 0], dtype=np.float32)  # Direct lane (BGR)
BLUE  = np.array([255, 0, 0], dtype=np.float32)  # Alternative lane (BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Resize with padding
    padded, pad_top, pad_left, new_h, new_w = letterbox(rgb, target_size=(320, 640))
    
    # 3. Normalize
    tensor = padded.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    # 5. Crop out padding to return to original proportions
    valid_pred = pred[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    full_pred = cv2.resize(valid_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # 6. Apply smooth overlay with NumPy (crash-safe)
    overlay = frame.copy()
    
    # Class 0: Direct Road (Green)
    mask_0 = (full_pred == 0)
    if np.any(mask_0):
        overlay[mask_0] = (0.5 * overlay[mask_0].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
        
    # Class 1: Alternative Road (Blue)
    mask_1 = (full_pred == 1)
    if np.any(mask_1):
        overlay[mask_1] = (0.5 * overlay[mask_1].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)
        
    # Class 2 (Background) remains completely unpainted
    
    out.write(overlay)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()
print(f"\nDone! Output saved to: {OUTPUT_PATH}")