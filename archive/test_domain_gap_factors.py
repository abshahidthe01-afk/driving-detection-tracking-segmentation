import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image
import os

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


# Load a clean BDD100K test image (known good baseline)
bdd_path = "bdd100k_drivable_subset/test/images/00000.png"
bdd_img = np.array(Image.open(bdd_path).convert("RGB"))
predict_and_report(bdd_img, "1. Original BDD100K PNG (clean)")

# Step 1: simulate JPEG/video-style compression artifacts on the SAME image
bdd_bgr = cv2.cvtColor(bdd_img, cv2.COLOR_RGB2BGR)
encode_success, encoded = cv2.imencode(".jpg", bdd_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
compressed_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
compressed_rgb = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)
predict_and_report(compressed_rgb, "2. Same BDD100K image, JPEG-compressed (quality=70)")

# Step 2: same image, resized down and back up (simulates our ffmpek downscale-then-model-resize pipeline)
small = cv2.resize(bdd_bgr, (640, 360), interpolation=cv2.INTER_AREA)
upscaled = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_LINEAR)
upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
predict_and_report(upscaled_rgb, "3. Same BDD100K image, downscaled+upscaled (simulates video pipeline)")

# Step 3: same image, brightness boosted to roughly match our video's higher brightness (~104 vs ~64 mean)
brightened = np.clip(bdd_img.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
predict_and_report(brightened, "4. Same BDD100K image, brightness boosted 1.6x (matches our video's brightness)")

print("\nIf test 1 looks fine but 2/3/4 break down, that specific factor (compression, resize, or brightness) is a real contributor to the domain gap.")