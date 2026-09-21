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
model.load_state_dict(torch.load("drivable_finetuned_v2.pth", map_location=device, weights_only=True))
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


# Test 1: solid gray image -- no scene content at all
gray = np.full((720, 1280, 3), 128, dtype=np.uint8)
predict_and_report(gray, "Solid gray image (no content)")

# Test 2: solid black
black = np.zeros((720, 1280, 3), dtype=np.uint8)
predict_and_report(black, "Solid black image")

# Test 3: solid white
white = np.full((720, 1280, 3), 255, dtype=np.uint8)
predict_and_report(white, "Solid white image")

# Test 4: random noise -- genuinely meaningless content
rng = np.random.RandomState(0)
noise = rng.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
predict_and_report(noise, "Random noise image")

print("\nIf the model confidently predicts mostly class 2 even on meaningless input,")
print("that indicates a learned class-imbalance bias/shortcut rather than genuine scene understanding.")