import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Pick sample 00001 (which we know has a road)
img_path = Path("bdd100k_drivable_subset/train/images/00001.png")
mask_path = Path("bdd100k_drivable_subset/train/masks/00001.png")

if not img_path.exists() or not mask_path.exists():
    print("Error: Could not find image or mask file.")
    exit()

# Load image (OpenCV loads BGR) and mask
img = cv2.imread(str(img_path))
mask = np.array(Image.open(mask_path))
if mask.ndim == 3:
    mask = mask[:, :, 0]

# Define display colors (BGR format)
GREEN = [0, 255, 0]      # Direct Road
BLUE = [255, 0, 0]       # Alternative Road
ORANGE = [0, 140, 255]   # Orange color from your old test

# 1. NEW / CORRECT MAPPING:
# Class 0 = Direct Road (Green), Class 1 = Alternative (Blue), Class 2 = Background (Uncolored)
view_correct = img.copy()
view_correct[mask == 0] = cv2.addWeighted(view_correct[mask == 0], 0.5, np.full_like(view_correct[mask == 0], GREEN), 0.5, 0)
view_correct[mask == 1] = cv2.addWeighted(view_correct[mask == 1], 0.5, np.full_like(view_correct[mask == 1], BLUE), 0.5, 0)

# 2. OLD MAPPING:
# Class 0 = Background (Uncolored), Class 1 = Direct (Green), Class 2 = Alternative (Orange)
view_old = img.copy()
view_old[mask == 1] = cv2.addWeighted(view_old[mask == 1], 0.5, np.full_like(view_old[mask == 1], GREEN), 0.5, 0)
view_old[mask == 2] = cv2.addWeighted(view_old[mask == 2], 0.5, np.full_like(view_old[mask == 2], ORANGE), 0.5, 0)

# Add title banners to both images
cv2.putText(view_correct, "CORRECT: 0=Road (Green), 2=Background (Clear)", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(view_old, "OLD: 2=Alternative (Orange painted on Sky/Buildings)", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2, cv2.LINE_AA)

# Stack top and bottom into one single comparison image
combined = np.vstack([view_correct, view_old])

output_filename = "verify_labels_visual.png"
cv2.imwrite(output_filename, combined)
print(f"\nSaved image comparison to: {output_filename}")
print("Open 'verify_labels_visual.png' in your project folder to check it.")