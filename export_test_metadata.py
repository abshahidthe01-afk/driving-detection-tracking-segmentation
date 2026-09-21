import os
import json
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Loading dataset...")
dataset = fouh.load_from_hub("dgural/bdd100k")

night_or_rainy = dataset.match(
    (fo.ViewField("timeofday.label") == "night") | (fo.ViewField("weather.label") == "rainy")
)
daytime_clear = dataset.match(
    (fo.ViewField("timeofday.label") == "daytime") & (fo.ViewField("weather.label") == "clear")
)
NUM_DAYTIME_TO_MIX_IN = min(len(daytime_clear), len(night_or_rainy))
daytime_subset = daytime_clear.take(NUM_DAYTIME_TO_MIX_IN, seed=42)
combined = night_or_rainy.concat(daytime_subset)
combined = combined.match(fo.ViewField("drivable.mask_path") != None)

# Reproduce the exact same shuffle and split as export_training_data.py
combined = combined.shuffle(seed=42)
n_total = len(combined)
n_val = max(1, int(n_total * 0.1))
n_test = max(1, int(n_total * 0.1))
test_view = combined[:n_test]

# Save metadata (weather, timeofday) for each test image, in the same order
# they were exported, so we can match them up by filename index later.
metadata = []
for i, sample in enumerate(test_view):
    metadata.append({
        "filename": f"{i:05d}.png",
        "weather": sample.weather.label if sample.weather else "unknown",
        "timeofday": sample.timeofday.label if sample.timeofday else "unknown",
    })

output_path = "bdd100k_drivable_subset/test/metadata.json"
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata for {len(metadata)} test images to {output_path}")