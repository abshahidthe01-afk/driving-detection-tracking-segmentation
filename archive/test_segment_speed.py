import torch
import numpy as np
import cv2
import time
import segmentation_models_pytorch as smp
import albumentations as A

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = "smp-hub/segformer-b0-640x1280-city-160k"
model = smp.from_pretrained(checkpoint).eval().to(device)
import albumentations as A

preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])
image_bgr = cv2.imread("test_frame.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
processed = preprocessing(image=image_rgb)["image"]
input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

# Warm-up runs (first run on GPU is always slower due to CUDA initialization -- don't count it)
with torch.no_grad():
    for _ in range(3):
        _ = model(input_tensor)

# Timed runs
torch.cuda.synchronize()  # make sure GPU finished before starting the clock
num_runs = 30
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        output = model(input_tensor)
torch.cuda.synchronize()  # make sure GPU finished before stopping the clock
end = time.time()

avg_time_ms = (end - start) / num_runs * 1000
fps = 1000 / avg_time_ms

print(f"Average inference time: {avg_time_ms:.2f} ms per frame")
print(f"Equivalent speed: {fps:.1f} FPS")