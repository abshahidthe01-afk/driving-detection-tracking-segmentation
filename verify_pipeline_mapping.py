import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

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

cap = cv2.VideoCapture("driving_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
success, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
processed = seg_preprocessing(image=frame_rgb)["image"]
input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    seg_output = seg_model(input_tensor)
pred_mask = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()

h, w = pred_mask.shape
print(f"Prediction shape: {pred_mask.shape}")

# Sample specific regions we KNOW the content of, based on typical dashcam framing:
# - Top ~30% of frame: almost always sky/buildings (NOT road)
# - Bottom-center ~20% of frame: almost always road directly ahead (IS road)
top_region = pred_mask[: int(h * 0.3), :]
bottom_center_region = pred_mask[int(h * 0.8):, int(w * 0.35):int(w * 0.65)]

print("\n--- TOP region (should be sky/buildings, i.e. NOT road) ---")
unique, counts = np.unique(top_region, return_counts=True)
for cls, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
    print(f"  class {cls}: {count/top_region.size*100:.1f}%")

print("\n--- BOTTOM-CENTER region (should be road directly ahead) ---")
unique, counts = np.unique(bottom_center_region, return_counts=True)
for cls, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
    print(f"  class {cls}: {count/bottom_center_region.size*100:.1f}%")

print("\nWhichever class dominates BOTTOM-CENTER but NOT TOP is the real 'road' class in THIS model's output.")