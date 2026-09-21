import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# 1. Load Model
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
except Exception:
    model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, classes=19)

model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0)
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# 2. Read frame 60 from the video (2 seconds in, when the car is moving)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read frame from '{VIDEO_PATH}'.")
    exit()

# Save the raw frame so you can see it
cv2.imwrite("debug_raw_frame.png", frame)

# 3. Letterbox function (target: 320 height x 640 width)
def letterbox(image, target_size=(320, 640)):
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, pad_top, pad_left, new_h, new_w

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
padded, pad_top, pad_left, new_h, new_w = letterbox(rgb, (320, 640))

# Save what the model actually "sees"
cv2.imwrite("debug_model_input.png", cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))

# 4. Normalize & Predict
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor = padded.astype(np.float32) / 255.0
tensor = (tensor - mean) / std
tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)
    # Convert logits into real percentage probabilities (0% to 100%)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = np.argmax(probs, axis=0)

# 5. Crop out padding to only analyze the real video area
valid_pred = pred[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
valid_probs = probs[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w]

print("\n--- DIAGNOSTIC RESULTS ---")
print("Classes predicted inside the video area:", np.unique(valid_pred))

p_road = valid_probs[0]  # Class 0: Direct Road
p_alt  = valid_probs[1]  # Class 1: Alternative Road
p_bg   = valid_probs[2]  # Class 2: Background

print(f"Highest confidence for Direct Road anywhere:      {p_road.max() * 100:.2f}%")
print(f"Highest confidence for Alternative Road anywhere: {p_alt.max() * 100:.2f}%")
print(f"Average confidence for Background across frame:   {p_bg.mean() * 100:.2f}%")

print("\nImages saved for inspection:")
print(" - 'debug_raw_frame.png' (original video frame)")
print(" - 'debug_model_input.png' (what the model saw after letterbox padding)")