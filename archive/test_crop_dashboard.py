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
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    unique, counts = np.unique(pred, return_counts=True)
    dominant_cls = unique[np.argmax(counts)]
    dominant_pct = counts.max() / pred.size * 100
    print(f"{label}: dominant class={dominant_cls} ({dominant_pct:.1f}%), classes present={list(unique)}")
    return pred


# Load the daytime frame (has visible dashboard/hood in lower portion)
cap = cv2.VideoCapture("driving_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
success, frame_bgr = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

predict_and_report(frame_rgb, "1. Full daytime frame (with dashboard visible)")

# Crop out the bottom ~25% (typically where dashboard/hood appears) and re-test
h, w, _ = frame_rgb.shape
cropped = frame_rgb[: int(h * 0.75), :, :]
predict_and_report(cropped, "2. Daytime frame, bottom 25% cropped (dashboard removed)")

# Same for night/rain frame, which has MORE dashboard visible (steering wheel, gauges)
cap = cv2.VideoCapture("driving_night_rain_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
success, frame_bgr2 = cap.read()
cap.release()
frame_rgb2 = cv2.cvtColor(frame_bgr2, cv2.COLOR_BGR2RGB)

predict_and_report(frame_rgb2, "3. Full night/rain frame (with dashboard visible)")

h2, w2, _ = frame_rgb2.shape
cropped2 = frame_rgb2[: int(h2 * 0.55), :, :]  # this clip has more dashboard, crop more
predict_and_report(cropped2, "4. Night/rain frame, bottom 45% cropped (dashboard removed)")