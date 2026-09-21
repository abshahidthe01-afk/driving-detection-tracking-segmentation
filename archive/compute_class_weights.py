import os
import numpy as np
from PIL import Image

mask_dir = "bdd100k_drivable_subset/train/masks"
filenames = os.listdir(mask_dir)
print(f"Scanning {len(filenames)} training masks...")

class_pixel_counts = np.zeros(3, dtype=np.int64)

for i, fname in enumerate(filenames):
    mask = np.array(Image.open(os.path.join(mask_dir, fname)))
    for cls in range(3):
        class_pixel_counts[cls] += (mask == cls).sum()

    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1}/{len(filenames)}")

total_pixels = class_pixel_counts.sum()
class_frequencies = class_pixel_counts / total_pixels

print("\n--- Class pixel frequencies across full training set ---")
class_names = ["background", "direct", "alternative"]
for name, count, freq in zip(class_names, class_pixel_counts, class_frequencies):
    print(f"  {name}: {count:,} pixels ({freq*100:.2f}%)")

# Standard inverse-frequency weighting: weight = 1 / frequency, then normalize
# so weights average to roughly 1 (keeps loss magnitude comparable to before).
inverse_freq_weights = 1.0 / class_frequencies
normalized_weights = inverse_freq_weights / inverse_freq_weights.sum() * 3

print("\n--- Corrected class weights (inverse frequency, normalized) ---")
for name, weight in zip(class_names, normalized_weights):
    print(f"  {name}: {weight:.4f}")

print(f"\nUse these as: class_weights = torch.tensor([{normalized_weights[0]:.4f}, {normalized_weights[1]:.4f}, {normalized_weights[2]:.4f}])")