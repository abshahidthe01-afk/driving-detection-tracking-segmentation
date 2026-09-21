import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

NUM_SEG_CLASSES = 3
model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, NUM_SEG_CLASSES, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0),
)
model.load_state_dict(torch.load("drivable_finetuned.pth", map_location=device, weights_only=True))
model = model.eval().to(device)

preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])

import os

# Grab an actual BDD100K test set image (one we evaluated well on) instead of a video frame
test_img_dir = "bdd100k_drivable_subset/test/images"
test_filenames = sorted(os.listdir(test_img_dir))
test_img_path = os.path.join(test_img_dir, test_filenames[0])
frame = cv2.imread(test_img_path)
print(f"Using BDD100K test image: {test_img_path}")
print(f"Frame shape: {frame.shape}")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
processed = preprocessing(image=frame_rgb)["image"]
input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
unique, counts = np.unique(pred_mask, return_counts=True)
print(f"\nPredicted class distribution (0=background, 1=direct, 2=alternative):")
for cls, count in zip(unique, counts):
    pct = count / pred_mask.size * 100
    print(f"  class {cls}: {count} pixels ({pct:.1f}%)")

# Save a visualization
color_map = {0: [0, 0, 0], 1: [0, 255, 0], 2: [0, 128, 255]}
vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
for cls, color in color_map.items():
    vis[pred_mask == cls] = color
vis_resized = cv2.resize(vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
overlay = cv2.addWeighted(frame, 0.5, vis_resized, 0.5, 0)
cv2.imwrite("debug_daytime_frame_overlay.jpg", overlay)
print("\nSaved debug_daytime_frame.jpg and debug_daytime_frame_overlay.jpg")