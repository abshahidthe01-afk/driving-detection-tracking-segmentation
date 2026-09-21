import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

seg_checkpoint = "smp-hub/segformer-b0-640x1280-city-160k"
seg_model = smp.from_pretrained(seg_checkpoint).eval().to(device)
seg_preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])
ROAD_CLASS_ID = 0

SEG_EVERY_N_FRAMES = 3  # must match the value used in pipeline.py


def get_road_mask(frame, width, height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = seg_preprocessing(image=frame_rgb)["image"]
    input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = seg_model(input_tensor)
    pred_mask = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()
    road_mask = (pred_mask == ROAD_CLASS_ID).astype(np.uint8)
    return cv2.resize(road_mask, (width, height), interpolation=cv2.INTER_NEAREST)


def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0  # both empty -- treat as perfect agreement
    return intersection / union


source_path = "driving_night_rain_720p.mp4"
cap = cv2.VideoCapture(source_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
last_reused_mask = None
iou_scores = []

while True:
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    # "Real" -- always fresh, every single frame (our reference)
    real_mask = get_road_mask(frame, width, height)

    # "Reused" -- mimics pipeline.py's frame-skipping behavior
    if frame_idx % SEG_EVERY_N_FRAMES == 1 or last_reused_mask is None:
        last_reused_mask = real_mask  # on a "fresh" frame, reused == real by definition
    reused_mask = last_reused_mask

    score = iou(real_mask, reused_mask)
    iou_scores.append(score)

    if frame_idx % 200 == 0:
        print(f"Frame {frame_idx}/{total_frames}, running avg IoU: {np.mean(iou_scores):.4f}")

cap.release()

iou_scores = np.array(iou_scores)
print("\n--- Results ---")
print(f"Mean IoU across {len(iou_scores)} frames: {iou_scores.mean():.4f}")
print(f"Min IoU (worst single frame): {iou_scores.min():.4f}")
print(f"Frames below 0.90 IoU: {(iou_scores < 0.90).sum()} ({(iou_scores < 0.90).mean()*100:.1f}%)")
print(f"Frames below 0.80 IoU: {(iou_scores < 0.80).sum()} ({(iou_scores < 0.80).mean()*100:.1f}%)")