import os
import shutil
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Loading dataset...")
dataset = fouh.load_from_hub("dgural/bdd100k")

# Our target: night OR rainy samples (the conditions where our pipeline struggled),
# plus a smaller number of clear daytime samples mixed in to avoid the model
# "forgetting" how to handle easy conditions (catastrophic forgetting).
night_or_rainy = dataset.match(
    (fo.ViewField("timeofday.label") == "night") | (fo.ViewField("weather.label") == "rainy")
)
daytime_clear = dataset.match(
    (fo.ViewField("timeofday.label") == "daytime") & (fo.ViewField("weather.label") == "clear")
)

print(f"Night/rainy samples available: {len(night_or_rainy)}")
print(f"Daytime/clear samples available: {len(daytime_clear)}")

# Take all night/rainy samples, and an equal-ish number of daytime/clear samples
# so the training set isn't overwhelmingly skewed toward the hard conditions only.
NUM_DAYTIME_TO_MIX_IN = min(len(daytime_clear), len(night_or_rainy))
daytime_subset = daytime_clear.take(NUM_DAYTIME_TO_MIX_IN, seed=42)

combined = night_or_rainy.concat(daytime_subset)
print(f"Combined training pool: {len(combined)} samples")

# Only keep samples that actually have a drivable-area mask (some may be missing it)
combined = combined.match(fo.ViewField("drivable.mask_path") != None)
print(f"Samples with valid drivable mask: {len(combined)}")

# Split into train (80%), val (10%), and test (10%)
# Val is used during training to monitor progress and pick the best checkpoint.
# Test is held out completely and only touched once, at the very end, to report
# final honest performance -- never used to make any training decisions.
combined = combined.shuffle(seed=42)
n_total = len(combined)
n_val = max(1, int(n_total * 0.1))
n_test = max(1, int(n_total * 0.1))
test_view = combined[:n_test]
val_view = combined[n_test:n_test + n_val]
train_view = combined[n_test + n_val:]
print(f"Train: {len(train_view)}, Val: {len(val_view)}, Test: {len(test_view)}")

# Export to disk as plain image + mask file pairs
output_root = "bdd100k_drivable_subset"
for split_name, view in [("train", train_view), ("val", val_view), ("test", test_view)]:
    img_dir = os.path.join(output_root, split_name, "images")
    mask_dir = os.path.join(output_root, split_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i, sample in enumerate(view):
        src_img = sample.filepath
        src_mask = sample.drivable.mask_path
        if src_mask is None or not os.path.exists(src_mask):
            continue
        basename = f"{i:05d}.png"
        shutil.copy(src_img, os.path.join(img_dir, basename))
        shutil.copy(src_mask, os.path.join(mask_dir, basename))

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] copied {i + 1}/{len(view)}")

    print(f"Done exporting {split_name}: {len(os.listdir(img_dir))} images, {len(os.listdir(mask_dir))} masks")

print(f"\nExport complete. Data is in ./{output_root}/")