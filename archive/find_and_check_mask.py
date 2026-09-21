import os
from pathlib import Path
from PIL import Image
import numpy as np

def find_mask_file():
    # Look in the current project directory and common FiftyOne download locations
    search_dirs = [
        Path("."),
        Path.home() / "fiftyone",
    ]
    
    print("Searching for mask files...")
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for ext in ("*.png", "*.npy"):
            for file_path in base_dir.rglob(ext):
                # Look for common mask keywords in path or filename
                path_lower = str(file_path).lower()
                if any(k in path_lower for k in ["mask", "drivable", "label", "target", "groundtruth"]):
                    return file_path
    return None

mask_path = find_mask_file()

if mask_path is None:
    print("\nCould not find a mask file automatically.")
    print("Please open this script and manually put the path to any mask file into 'mask_path'.")
    exit()

print(f"\nFound mask file at: {mask_path}")

# Load the mask
if str(mask_path).endswith(".npy"):
    mask = np.load(mask_path)
else:
    # Use PIL to get the raw integer class IDs
    mask = np.array(Image.open(mask_path))

# If mask has 3 channels, take the first channel
if mask.ndim == 3:
    mask = mask[:, :, 0]

print("\n--- RESULTS ---")
print("All unique class IDs in this mask:", np.unique(mask))

# Check the sky (top 30 rows)
sky_pixels = mask[:30, :]
sky_id = np.bincount(sky_pixels.flatten()).argmax()
print(f"Most frequent ID in the SKY (top of image): {sky_id}")

# Check the road area (bottom-center of image)
h, w = mask.shape
road_pixels = mask[int(h * 0.7) : int(h * 0.9), int(w * 0.3) : int(w * 0.7)]
road_id = np.bincount(road_pixels.flatten()).argmax()
print(f"Most frequent ID in the ROAD (bottom-center): {road_id}")