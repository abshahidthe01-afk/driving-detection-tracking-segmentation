import os
from pathlib import Path

base_path = Path("bdd100k_drivable_subset")

if not base_path.exists():
    print("Folder 'bdd100k_drivable_subset' not found!")
    exit()

print(f"Inspecting directory structure of: {base_path.resolve()}\n")

# List directories and sample files
for root, dirs, files in os.walk(base_path):
    depth = root.replace(str(base_path), "").count(os.sep)
    if depth > 3:
        continue
    indent = " " * 4 * depth
    folder_name = os.path.basename(root)
    print(f"{indent}[Folder] {folder_name}/ (contains {len(files)} files)")
    
    # Print the first 3 files as examples
    for f in files[:3]:
        print(f"{indent}   ├── {f}")
    if len(files) > 3:
        print(f"{indent}   └── ... and {len(files) - 3} more")