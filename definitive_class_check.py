import numpy as np
from PIL import Image, ImageDraw
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Loading dataset...")
dataset = fouh.load_from_hub("dgural/bdd100k")

# Find a sample that actually has polylines with drivable-area labels
sample = None
for s in dataset:
    if s.polylines is not None and len(s.polylines.polylines) > 0:
        labels = [p.label for p in s.polylines.polylines]
        if "direct" in labels or "alternative" in labels:
            sample = s
            break

if sample is None:
    print("No sample found with direct/alternative polyline labels in the 'polylines' field.")
else:
    print(f"Found sample: {sample.filepath}")
    print(f"Image dimensions needed for polyline coordinate conversion...")

    img = Image.open(sample.filepath)
    img_w, img_h = img.size
    print(f"Image size: {img_w}x{img_h}")

    mask = np.array(Image.open(sample.drivable.mask_path))
    mask_h, mask_w = mask.shape
    print(f"Mask size: {mask_w}x{mask_h}")

    for polyline in sample.polylines.polylines:
        if polyline.label not in ("direct", "alternative"):
            continue

        print(f"\n--- Polyline labeled '{polyline.label}' ---")
        # Polyline points are normalized (0-1) relative to image dimensions.
        # Convert to mask pixel coordinates and check what value appears there.
        for shape in polyline.points:
            # Convert normalized points to actual mask pixel coordinates
            pixel_points = [(x * mask_w, y * mask_h) for x, y in shape]

            # Create a blank mask, draw this polygon filled, then find where
            # it overlaps with the actual drivable mask to sample values.
            poly_mask_img = Image.new("L", (mask_w, mask_h), 0)
            draw = ImageDraw.Draw(poly_mask_img)
            draw.polygon(pixel_points, fill=1)
            poly_mask = np.array(poly_mask_img)

            # Sample what values appear in the real mask, within this polygon's area
            values_inside = mask[poly_mask == 1]
            if len(values_inside) == 0:
                print("  (polygon produced no pixels -- skipping)")
                continue
            unique, counts = np.unique(values_inside, return_counts=True)
            print(f"  Mask values found inside this '{polyline.label}' polygon:")
            for val, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
                pct = count / len(values_inside) * 100
                print(f"    mask value {val}: {pct:.1f}% of polygon area ({count} pixels)")