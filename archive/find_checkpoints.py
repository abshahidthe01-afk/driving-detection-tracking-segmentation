from pathlib import Path

print("Searching for saved model files in your project folder...\n")

extensions = ("*.pth", "*.pt", "*.ckpt", "*.bin", "*.tar")
found_files = []

for ext in extensions:
    found_files.extend(Path(".").rglob(ext))

if not found_files:
    print("No checkpoint files (.pth, .pt, .ckpt) were found in this directory.")
else:
    print(f"Found {len(found_files)} saved checkpoint file(s):\n")
    for idx, path in enumerate(found_files, start=1):
        print(f"  [{idx}] {path}")