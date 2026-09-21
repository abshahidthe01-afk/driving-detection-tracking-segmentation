import cv2
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

# 1. Load Model
CHECKPOINT_PATH = "drivable_finetuned.pth"
VIDEO_PATH = "driving_night_rain_720p.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 2. Grab frame 60
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video frame.")
    exit()

h, w = frame.shape[:2]

# 3. Method 1: Crop the windshield (cut out top ceiling and lower dashboard)
# In portrait video, road is in the middle section
crop_top = int(h * 0.25)
crop_bottom = int(h * 0.75)
cropped_frame = frame[crop_top:crop_bottom, :]

# 4. Method 2: Brightness boost using CLAHE (adaptive contrast)
def boost_lighting(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_boosted = clahe.apply(l)
    enhanced = cv2.merge((l_boosted, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

boosted_cropped = boost_lighting(cropped_frame)

# Save visual checks
cv2.imwrite("test_cropped_normal.png", cropped_frame)
cv2.imwrite("test_cropped_boosted.png", boosted_cropped)

# Helper function to evaluate an image
def evaluate_image(bgr_img, name):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 320))
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    tensor = torch.tensor(tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(probs, axis=0)
        
    p_road = probs[0]  # Class 0: Direct Road
    p_alt  = probs[1]  # Class 1: Alternative Road
    classes_found = np.unique(pred)
    
    print(f"\n[{name}]")
    print(f"  Classes detected: {classes_found}")
    print(f"  Max Direct Road confidence:      {p_road.max() * 100:.2f}%")
    print(f"  Max Alternative Road confidence: {p_alt.max() * 100:.2f}%")
    
    # Save segmentation preview
    overlay = resized.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay[pred == 0] = (0.5 * overlay[pred == 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[pred == 1] = (0.5 * overlay[pred == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.imwrite(f"result_{name}.png", overlay)

# Run tests
evaluate_image(cropped_frame, "cropped_normal")
evaluate_image(boosted_cropped, "cropped_boosted")

print("\nPreviews saved: 'result_cropped_normal.png' and 'result_cropped_boosted.png'")