import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from ultralytics import YOLO

# --- 1. SETTINGS & PATHS ---
SEG_CHECKPOINT = "drivable_finetuned.pth"
YOLO_CHECKPOINT = "yolov8s.pt"

# Using your verified daytime video from the list
VIDEO_PATH = "driving_720p.mp4"
OUTPUT_PATH = "final_cv_pipeline_output.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. LOAD SEGMENTATION MODEL (SegFormer) ---
print("Loading SegFormer model...")
try:
    seg_model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
except Exception:
    seg_model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, classes=19)

seg_model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0)
)

checkpoint = torch.load(SEG_CHECKPOINT, map_location=device, weights_only=False)
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        seg_model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        seg_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        seg_model.load_state_dict(checkpoint)
else:
    seg_model.load_state_dict(checkpoint)

seg_model.to(device)
seg_model.eval()

# --- 3. LOAD OBJECT DETECTOR (YOLOv8) ---
print("Loading YOLOv8 detector...")
yolo_model = YOLO(YOLO_CHECKPOINT)

# --- 4. VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video: {VIDEO_PATH}")
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

GREEN = np.array([0, 255, 0], dtype=np.float32)  # Ego Lane
BLUE  = np.array([255, 0, 0], dtype=np.float32)  # Side Lanes

# Common driving object classes (person, bike, car, motorcycle, bus, truck)
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- A. RUN SEGMENTATION ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_seg = cv2.resize(rgb, (640, 320))

    tensor = resized_seg.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_output = seg_model(tensor)
        pred = torch.argmax(seg_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Blend lane colors
    composite = frame.copy()
    mask_ego = (pred_full == 0)
    mask_alt = (pred_full == 1)

    if np.any(mask_ego):
        composite[mask_ego] = (0.5 * composite[mask_ego].astype(np.float32) + 0.5 * GREEN).astype(np.uint8)
    if np.any(mask_alt):
        composite[mask_alt] = (0.5 * composite[mask_alt].astype(np.float32) + 0.5 * BLUE).astype(np.uint8)

    # --- B. RUN YOLO DETECTION ---
    results = yolo_model(frame, classes=TARGET_CLASSES, conf=0.35, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = yolo_model.names[cls_id]

        # Draw yellow bounding box
        cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw label tag above box
        label = f"{class_name} {conf:.2f}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(composite, (x1, y1 - 22), (x1 + w_text, y1), (0, 255, 255), -1)
        cv2.putText(composite, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    out.write(composite)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()
print(f"\nDone! Saved combined output to: {OUTPUT_PATH}")