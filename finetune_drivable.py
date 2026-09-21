import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

DATA_ROOT = "bdd100k_drivable_subset"
NUM_CLASSES = 3  # background, direct, alternative
INPUT_HEIGHT, INPUT_WIDTH = 320, 640  # matches our chosen inference resolution
BATCH_SIZE = 4  # kept small given 4GB VRAM
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "drivable_finetuned_v2.pth"  # new filename -- keep v1 as a documented, informative failure case


class DrivableDataset(Dataset):
    def __init__(self, split, transform):
        self.img_dir = os.path.join(DATA_ROOT, split, "images")
        self.mask_dir = os.path.join(DATA_ROOT, split, "masks")
        self.filenames = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname)))

        transformed = self.transform(image=image, mask=mask)
        image_t = torch.as_tensor(transformed["image"]).permute(2, 0, 1)
        mask_t = torch.as_tensor(transformed["mask"], dtype=torch.long)
        return image_t, mask_t


# Same normalization as the original checkpoint, but at our chosen (smaller) resolution
train_transform = A.Compose([
    A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),  # simulates varied video compression
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])
val_transform = A.Compose([
    A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
    A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1.0),
])

train_dataset = DrivableDataset("train", train_transform)
val_dataset = DrivableDataset("val", val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Load our existing pretrained Cityscapes checkpoint, then replace its output head
# to predict our 3 classes instead of Cityscapes' 19.
print("Loading pretrained checkpoint...")
model = smp.from_pretrained("smp-hub/segformer-b0-640x1280-city-160k")

# Replace the segmentation head to output 3 classes instead of 19.
# We keep the pretrained encoder (the part that "understands" images generally)
# and only reset the final classification layer. Importantly, we must preserve
# the original head's upsampling step (scale_factor=4.0), or the output stays
# at 1/4 resolution and won't match our full-resolution masks.
old_head = model.segmentation_head
print(f"Original segmentation head: {old_head}")

model.segmentation_head = nn.Sequential(
    nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
    nn.UpsamplingBilinear2d(scale_factor=4.0),
)
model = model.to(device)

# Class-weighted loss to address the severe imbalance we measured directly:
# background=12.28%, direct=4.95%, alternative=82.77% of training pixels.
# Weights computed via inverse frequency (see compute_class_weights.py) -- this
# corrects our earlier flawed weighting, which left the dominant "alternative"
# class at full weight and inadvertently let the model learn a majority-class
# shortcut (confirmed via meaningless-input testing, see PROJECT_LOG).
class_weights = torch.tensor([0.8265, 2.0509, 0.1226]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, batch {batch_idx+1}/{len(train_loader)}, loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} done. Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f} ---")

    # Save checkpoint if this is the best validation performance so far.
    # This protects against losing progress if a later epoch overfits or the run is interrupted.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  New best val loss -- saved checkpoint to {CHECKPOINT_PATH}")

print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved at: {CHECKPOINT_PATH}")