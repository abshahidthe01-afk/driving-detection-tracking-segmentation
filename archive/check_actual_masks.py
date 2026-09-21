from pathlib import Path
from PIL import Image
import numpy as np

# Let's inspect the first 5 training masks to be completely sure
mask_dir = Path("bdd100k_drivable_subset/train/masks")
mask_files = sorted(list(mask_dir.glob("*.png")))[:5]

print(f"Found {len(mask_files)} sample masks to check.\n")

for i, file_path in enumerate(mask_files):
    # Open the raw mask image
    mask = np.array(Image.open(file_path))
    
    # If it has 3 color channels, take just the first channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]
        
    unique_ids = np.unique(mask)
    
    # Top 20% of the image (Sky / Background area)
    top_area = mask[: int(mask.shape[0] * 0.2), :]
    sky_id = np.bincount(top_area.flatten()).argmax()
    
    # Bottom-center area (Road directly in front of the vehicle)
    h, w = mask.shape
    road_area = mask[int(h * 0.7) : int(h * 0.9), int(w * 0.3) : int(w * 0.7)]
    road_id = np.bincount(road_area.flatten()).argmax()
    
    print(f"Mask {i} ({file_path.name}):")
    print(f"  All class IDs present: {unique_ids}")
    print(f"  Dominant ID in Sky (top 20%): {sky_id}")
    print(f"  Dominant ID in Road (bottom-center): {road_id}\n")