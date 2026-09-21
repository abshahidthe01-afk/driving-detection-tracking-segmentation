import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pretrained Cityscapes checkpoint
checkpoint = "smp-hub/segformer-b0-640x1280-city-160k"
model = smp.from_pretrained(checkpoint).eval().to(device)
preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])
# Load our test frame (OpenCV loads as BGR, model expects RGB)
image_bgr = cv2.imread("test_frame.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Preprocess
processed = preprocessing(image=image_rgb)["image"]
input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Output shape: [1, num_classes, H, W] -- take the class with highest score per pixel
pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

print(f"Output mask shape: {pred_mask.shape}")
print(f"Unique class IDs present: {np.unique(pred_mask)}")

# Cityscapes class ID for "road" is 0
ROAD_CLASS_ID = 0
road_mask = (pred_mask == ROAD_CLASS_ID).astype(np.uint8) * 255

# Resize mask back to original image size and save as a visual overlay
road_mask_resized = cv2.resize(road_mask, (image_bgr.shape[1], image_bgr.shape[0]))
overlay = image_bgr.copy()
overlay[road_mask_resized == 255] = [0, 255, 0]  # green over road pixels
blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

cv2.imwrite("test_frame_road_overlay.jpg", blended)
print("Saved test_frame_road_overlay.jpg")