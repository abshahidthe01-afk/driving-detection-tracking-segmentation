import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: inspect the raw checkpoint file directly, before loading it into any model
checkpoint = torch.load("drivable_finetuned.pth", map_location=device, weights_only=True)
print(f"Checkpoint type: {type(checkpoint)}")
print(f"Number of keys in checkpoint: {len(checkpoint)}")
print("First 5 keys:", list(checkpoint.keys())[:5])
print("Last 5 keys:", list(checkpoint.keys())[-5:])

# Check the segmentation head weights specifically -- these are the ones that
# should differ most from the original pretrained model, since we replaced and trained them.
head_keys = [k for k in checkpoint.keys() if "segmentation_head" in k]
print(f"\nSegmentation head keys found: {head_keys}")
for k in head_keys:
    tensor = checkpoint[k]
    print(f"  {k}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")

# Step 2: build the model architecture and attempt to load, checking for mismatches explicitly
print("\n--- Attempting load_state_dict with strict=True (will error on any mismatch) ---")
model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")
model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, 3, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0),
)

# Print the model's OWN segmentation head keys, to compare against the checkpoint's keys
model_head_keys = [k for k in model.state_dict().keys() if "segmentation_head" in k]
print(f"Model's own segmentation head keys: {model_head_keys}")

missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"\nMissing keys (in model but not checkpoint): {missing}")
print(f"Unexpected keys (in checkpoint but not model): {unexpected}")