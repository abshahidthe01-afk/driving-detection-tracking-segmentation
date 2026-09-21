from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.track(
    source="driving_720p.mp4",
    save=True,
    device=0,
    imgsz=640,
    tracker="bytetrack.yaml",   # built-in ByteTrack config
    persist=True                 # keep track IDs consistent across frames
)

print("Done. Check the 'runs/detect/track' folder for output.")