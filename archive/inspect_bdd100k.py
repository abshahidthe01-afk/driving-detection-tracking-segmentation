import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Loading dataset (this may take a while on first run -- downloading from Hugging Face)...")
dataset = fouh.load_from_hub("dgural/bdd100k")

print("\n--- Dataset summary ---")
print(dataset)

print("\n--- Sample fields ---")
print(dataset.get_field_schema())

print("\n--- First sample (full detail) ---")
sample = dataset.first()
print(sample)