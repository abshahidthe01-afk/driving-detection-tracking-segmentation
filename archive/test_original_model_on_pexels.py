import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the ORIGINAL, unmodified 19-class Cityscapes model -- no fine-tuning, no head replacement
model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k").eval().to(device)
preprocessing = A.Compose([
    A.Resize(height=320, width=640),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])

def test_frame(video_path, frame_num, label):
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
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    unique, counts = np.unique(pred, return_counts=True)
    print(f"\n{label} ({video_path}, frame {frame_num}):")
    print(f"  Number of distinct classes predicted: {len(unique)}")
    for cls, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
        pct = count / pred.size * 100
        print(f"  class {cls}: {pct:.1f}%")

    # Specifically check "road" (class 0) presence -- the class we cared about originally
    road_pct = (pred == 0).sum() / pred.size * 100
    print(f"  --> 'road' (class 0) coverage: {road_pct:.1f}%")

    # Save visual overlay for inspection
    color_map = np.random.RandomState(42).randint(0, 255, (19, 3), dtype=np.uint8)
    color_map[0] = [0, 255, 0]  # force road to bright green for easy visual check
    vis = color_map[pred]
    vis_resized = cv2.resize(vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(frame, 0.5, vis_resized, 0.5, 0)
    out_name = f"debug_original_model_{label.replace(' ', '_')}.jpg"
    cv2.imwrite(out_name, overlay)
    print(f"  Saved {out_name}")


test_frame("driving_720p.mp4", 200, "daytime")
test_frame("driving_night_rain_720p.mp4", 200, "night_rain")