from pathlib import Path

print("Searching for video files in your project directory...\n")

video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
found_videos = []

for ext in video_extensions:
    found_videos.extend(Path(".").glob(ext))

if not found_videos:
    print("No video files found in this folder.")
else:
    print("Found the following video files:")
    for v in found_videos:
        print(f"  - {v.name}")