import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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


def test_video_frame(video_path, frame_num, label):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    if not success:
        print(f"Could not read frame from {video_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = preprocessing(image=frame_rgb)["image"]
    input_tensor = torch.as_tensor(processed).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"\n{label}: shape={frame.shape}")
    for cls, count in zip(unique, counts):
        pct = count / pred_mask.size * 100
        print(f"  class {cls}: {pct:.1f}%")

    color_map = {0: [0, 0, 0], 1: [0, 255, 0], 2: [0, 128, 255]}
    vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        vis[pred_mask == cls] = color
    vis_resized = cv2.resize(vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(frame, 0.5, vis_resized, 0.5, 0)
    out_name = f"debug_v2_{label.replace(' ', '_').replace('/', '_')}.jpg"
    cv2.imwrite(out_name, overlay)
    print(f"  Saved {out_name}")


test_video_frame("driving_720p.mp4", 200, "daytime")
test_video_frame("driving_night_rain_720p.mp4", 200, "night_rain")