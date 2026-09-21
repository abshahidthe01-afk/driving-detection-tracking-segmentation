import fiftyone as fo
import fiftyone.utils.huggingface as fouh

dataset = fouh.load_from_hub("dgural/bdd100k")

print(f"Total samples: {len(dataset)}")

# Count by timeofday
print("\n--- Counts by timeofday ---")
for label in ["daytime", "night", "dawn/dusk", "undefined"]:
    count = len(dataset.match({"timeofday.label": label}))
    print(f"{label}: {count}")

# Count by weather
print("\n--- Counts by weather ---")
for label in ["clear", "overcast", "rainy", "snowy", "partly cloudy", "foggy", "undefined"]:
    count = len(dataset.match({"weather.label": label}))
    print(f"{label}: {count}")

# The intersection we actually care about: night OR rainy (our failure case)
night_or_rainy = dataset.match(
    (fo.ViewField("timeofday.label") == "night") | (fo.ViewField("weather.label") == "rainy")
)
print(f"\nNight OR rainy samples: {len(night_or_rainy)}")

night_and_rainy = dataset.match(
    (fo.ViewField("timeofday.label") == "night") & (fo.ViewField("weather.label") == "rainy")
)
print(f"Night AND rainy samples: {len(night_and_rainy)}")