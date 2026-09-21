import numpy as np
from PIL import Image
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Loading dataset...")
dataset = fouh.load_from_hub("dgural/bdd100k")

# Find a sample and inspect its drivable field directly -- both the mask pixel
# values AND the explicit polyline labels, so we can match them up definitively.
sample = dataset.first()

print("\n--- Polyline labels (ground truth, human-readable) ---")
for polyline in sample.drivable.polylines if hasattr(sample.drivable, "polylines") else []:
    print(f"  label: {polyline.label}, areaType: {polyline.attributes}")

print("\n--- Drivable field structure ---")
print(sample.drivable)

# Load the actual mask file and check pixel values
mask_path = sample.drivable.mask_path
mask = np.array(Image.open(mask_path))
print(f"\n--- Mask file: {mask_path} ---")
print(f"Unique pixel values: {np.unique(mask)}")
for val in np.unique(mask):
    count = (mask == val).sum()
    pct = count / mask.size * 100
    print(f"  value {val}: {count} pixels ({pct:.1f}%)")

# Cross-reference: fiftyone's Segmentation field should have a documented
# convention. Let's check if there's a mask_targets or similar metadata.
print("\n--- Dataset-level metadata for 'drivable' field ---")
field_schema = dataset.get_field_schema()
print(field_schema.get("drivable"))

# Also check dataset.info or mask_targets, which fiftyone sometimes stores
# for segmentation fields to document the class mapping explicitly.
print("\n--- Dataset info (may contain class mapping) ---")
print(dataset.info)

print("\n--- Dataset mask_targets (if defined) ---")
print(dataset.mask_targets if hasattr(dataset, "mask_targets") else "No mask_targets attribute")