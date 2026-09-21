import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

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


def predict_and_report(image_rgb, label):
    processed = preprocessing(image=image_rgb)["image"]
    input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    max_class0_conf = probs[0, 0].max().item()  # "direct" (ego lane) per corrected mapping
    max_class1_conf = probs[0, 1].max().item()  # "alternative"
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    unique, counts = np.unique(pred, return_counts=True)
    print(f"\n{label}:")
    print(f"  Max softmax confidence -- class 0 (direct): {max_class0_conf*100:.2f}%, class 1 (alternative): {max_class1_conf*100:.2f}%")
    for cls, count in zip(unique, counts):
        pct = count / pred.size * 100
        print(f"  class {cls}: {pct:.1f}% of pixels")


cap = cv2.VideoCapture("driving_night_rain_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)  # matching the other report's stated "frame 60"
success, frame_bgr = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = frame_rgb.shape
print(f"Frame shape: {frame_rgb.shape}")

predict_and_report(frame_rgb, "1. Full frame, no crop (letterboxed by aspect-preserving resize)")

# Proposed windshield ROI: 0.15 to 0.55 normalized height
y1, y2 = int(h * 0.15), int(h * 0.55)
roi = frame_rgb[y1:y2, :, :]
print(f"\nROI crop shape: {roi.shape}")
predict_and_report(roi, "2. Windshield ROI crop (0.15-0.55 normalized height)")