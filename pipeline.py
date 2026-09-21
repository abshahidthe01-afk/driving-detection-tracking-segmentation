import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import albumentations as A
import segmentation_models_pytorch as smp
from ultralytics import YOLO

# ---------- Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Detection + tracking model
yolo_model = YOLO("yolov8s.pt")

# Segmentation model -- fine-tuned drivable-area model (3 classes: direct/alternative/background),
# fine-tuned from the original Cityscapes checkpoint on BDD100K night/rain + daytime data.
# Using v1 checkpoint: verified via held-out test mIoU (0.75 overall, 0.80 on night) and
# confirmed correct label mapping (0=direct, 1=alternative, 2=background) via polygon cross-check.
# See PROJECT_LOG for full training/debugging history.
NUM_SEG_CLASSES = 3
seg_model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
seg_model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0),
)
seg_model.load_state_dict(torch.load("drivable_finetuned.pth", map_location=device, weights_only=True))
seg_model = seg_model.eval().to(device)

seg_preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])
# Class mapping verified via direct polygon-to-pixel cross-check against BDD100K's
# explicit labels: 0=direct (ego lane), 1=alternative (adjacent lane), 2=background.
# See PROJECT_LOG for full verification history.

# Run segmentation only every Nth frame, reuse last mask in between (segmentation is the dominant cost)
SEG_EVERY_N_FRAMES = 3
last_direct_mask_resized = None
last_alternative_mask_resized = None

# ---------- Video setup ----------
source_path = "driving_night_rain_720p.mp4"
cap = cv2.VideoCapture(source_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "pipeline_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing {total_frames} frames at {width}x{height}, {fps:.1f} FPS source")

# ---------- Main loop ----------
frame_idx = 0
timings = {"read": 0.0, "seg": 0.0, "track": 0.0, "plot": 0.0, "write": 0.0}

while True:
    t0 = time.time()
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1
    t1 = time.time()
    timings["read"] += t1 - t0

    # --- Segmentation (run every Nth frame, reuse last mask otherwise) ---
    if frame_idx % SEG_EVERY_N_FRAMES == 1 or last_direct_mask_resized is None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = seg_preprocessing(image=frame_rgb)["image"]
        input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_output = seg_model(input_tensor)
        pred_mask = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Keep direct (0) and alternative (1) as separate masks so we can color them differently.
        direct_mask = (pred_mask == 0).astype(np.uint8) * 255
        alternative_mask = (pred_mask == 1).astype(np.uint8) * 255
        last_direct_mask_resized = cv2.resize(direct_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        last_alternative_mask_resized = cv2.resize(alternative_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    overlay = frame.copy()
    overlay[last_direct_mask_resized == 255] = [0, 255, 0]        # green: current/ego lane (direct)
    overlay[last_alternative_mask_resized == 255] = [255, 0, 0]   # blue: adjacent lane (alternative)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    timings["seg"] += t2 - t1

    # --- Detection + Tracking ---
    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=0 if device == "cuda" else "cpu",
        verbose=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()
    timings["track"] += t3 - t2

    annotated_frame = results[0].plot(img=frame)
    t4 = time.time()
    timings["plot"] += t4 - t3

    writer.write(annotated_frame)
    t5 = time.time()
    timings["write"] += t5 - t4

    if frame_idx % 100 == 0:
        print(f"Processed frame {frame_idx}/{total_frames}")

cap.release()
writer.release()
print(f"Done. Output saved to {output_path}")

print("\n--- Timing breakdown (average ms/frame) ---")
for stage, total in timings.items():
    avg_ms = (total / frame_idx) * 1000
    print(f"{stage:>8}: {avg_ms:6.2f} ms/frame")
total_avg_ms = sum(timings.values()) / frame_idx * 1000
print(f"{'TOTAL':>8}: {total_avg_ms:6.2f} ms/frame  ({1000/total_avg_ms:.1f} FPS)")
