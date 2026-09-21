import fiftyone as fo
import numpy as np

# 1. Check all available datasets in FiftyOne
datasets = fo.list_datasets()
print(f"Available FiftyOne datasets: {datasets}")

if not datasets:
    print("No FiftyOne datasets found! If your masks are saved in a folder, load a PNG directly with cv2 or PIL.")
    exit()

# Pick the BDD100k dataset
ds_name = [d for d in datasets if "bdd" in d.lower()][0] if any("bdd" in d.lower() for d in datasets) else datasets[0]
print(f"Inspecting dataset: {ds_name}\n")
dataset = fo.load_dataset(ds_name)

# 2. Grab the first sample
sample = dataset.first()

# 3. Find the segmentation mask inside the sample
mask = None
for field_name in sample.field_names:
    val = sample[field_name]
    if hasattr(val, "mask") and val.mask is not None:
        mask = val.mask
        print(f"Found mask in field: '{field_name}'")
        break

if mask is None:
    print("Could not find a mask field automatically in this sample.")
    exit()

# Convert mask to integer array
mask = np.array(mask, dtype=int)

print("\n--- RESULTS ---")
print("All unique ID numbers in this mask:", np.unique(mask))

# Check top 30 rows (where the sky usually is)
sky_pixels = mask[:30, :]
sky_id = np.bincount(sky_pixels.flatten()).argmax()
print(f"Most frequent ID in the SKY (top of image): {sky_id}")

# Check bottom-center area (where the road right in front of the car usually is)
h, w = mask.shape
road_pixels = mask[int(h * 0.7) : int(h * 0.9), int(w * 0.3) : int(w * 0.7)]
road_id = np.bincount(road_pixels.flatten()).argmax()
print(f"Most frequent ID in the ROAD (bottom-center): {road_id}")