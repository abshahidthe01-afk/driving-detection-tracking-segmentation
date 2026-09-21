import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# --- 1. SETTINGS ---
CHECKPOINT_PATH = "drivable_finetuned.pth"

# Put your landscape dashcam video filename here:
VIDEO_PATH = "driving_720p.mp4" 
OUTPUT_PATH = "external_dashcam_segmented.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. LOAD MODEL ---
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

# --- 3. PROCESS VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open '{VIDEO_PATH}'. Please set the correct file name.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

print(f"Processing '{VIDEO_PATH}' ({total_frames} frames)...")

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GREEN = np.array([0, 255, 0], dtype=np.float32)  # Ego Lane (Direct)
BLUE  = np.array([255, 0, 0], dtype=np.float32)  # Side Lanes (Alternative)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 320))
    
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    overlay = frame.copy()
    mask_ego = (pred_full == 0)
    mask_alt = (pred_full == 1)
    
    if np.any(mask_ego):
        overlay[mask_ego] = (0.5 * overlay[mask_ego].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
    if np.any(mask_alt):
        overlay[mask_alt] = (0.5 * overlay[mask_alt].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)
        
    out.write(overlay)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()
print(f"\nDone! Saved output to '{OUTPUT_PATH}'.")