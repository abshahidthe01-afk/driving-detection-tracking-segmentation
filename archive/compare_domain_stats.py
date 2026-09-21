import numpy as np
from PIL import Image
import cv2
import os

def print_stats(name, img_rgb):
    print(f"\n{name}")
    print(f"  shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
    print(f"  overall mean: {img_rgb.mean():.2f}, std: {img_rgb.std():.2f}")
    print(f"  min/max: {img_rgb.min()} / {img_rgb.max()}")
    for i, channel in enumerate(["R", "G", "B"]):
        print(f"  {channel} channel: mean={img_rgb[:,:,i].mean():.2f}, std={img_rgb[:,:,i].std():.2f}")

# BDD100K test image (known good result)
bdd_path = "bdd100k_drivable_subset/test/images/00000.png"
bdd_img = np.array(Image.open(bdd_path).convert("RGB"))
print_stats(f"BDD100K test image ({bdd_path})", bdd_img)

# Our daytime video frame
cap = cv2.VideoCapture("driving_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
success, frame_bgr = cap.read()
cap.release()
daytime_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
print_stats("Our daytime video frame (driving_720p.mp4)", daytime_rgb)

# Our night/rain video frame
cap = cv2.VideoCapture("driving_night_rain_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
success, frame_bgr = cap.read()
cap.release()
night_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
print_stats("Our night/rain video frame (driving_night_rain_720p.mp4)", night_rgb)