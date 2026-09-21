import numpy as np
from PIL import Image
import os

mask_dir = "bdd100k_drivable_subset/train/masks"
sample_files = sorted(os.listdir(mask_dir))[:5]

for fname in sample_files:
    path = os.path.join(mask_dir, fname)
    mask = np.array(Image.open(path))
    unique_values, counts = np.unique(mask, return_counts=True)
    print(f"{fname}: shape={mask.shape}, dtype={mask.dtype}")
    print(f"  unique pixel values: {unique_values}")
    print(f"  counts: {counts}")
    print()