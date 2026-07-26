from collections import Counter
from pathlib import Path

from PIL import Image


def remove_wrong_sized_images(dataset_dir: str, auto_delete: bool = False):
    base_dir = Path(dataset_dir)
    if not base_dir.exists():
        print(f"[error] Directory {base_dir.resolve()} does not exist.")
        return

    print(f"Scanning {base_dir.resolve()} for image sizes...\n")
    
    # Recursively find all PNG files in the dataset folder
    all_images = list(base_dir.rglob("*.png"))
    
    if not all_images:
        print("No PNG images found in the specified directory.")
        return

    size_counts = Counter()
    image_sizes = {}

    for img_path in all_images:
        try:
            # PIL lazily loads the image header, so this is very fast and memory efficient
            with Image.open(img_path) as img:
                sz = img.size  # Returns a tuple (width, height)
                size_counts[sz] += 1
                image_sizes[img_path] = sz
        except Exception as e:
            print(f"[warning] Could not read {img_path.name}: {e}")

    if not size_counts:
        return

    # Determine the "correct" standard size based on the majority
    expected_size = size_counts.most_common(1)[0][0]
    expected_count = size_counts[expected_size]
    
    print("==================================================")
    print(f"Majority Size Detected: {expected_size[0]}x{expected_size[1]} pixels")
    print(f"Images matching this size: {expected_count}")
    print("==================================================\n")

    # Find and  delete the outliers
    outliers = []
    for img_path, sz in image_sizes.items():
        if sz != expected_size:
            outliers.append((img_path, sz))

    if not outliers:
        print("[success] All images in the dataset match the expected standard size!")
        return

    print(f"Found {len(outliers)} images with non-standard sizes:\n")
    
    for file_path, sz in outliers:
        if auto_delete:
            try:
                file_path.unlink()
                print(f"[DELETED] {file_path.name} (Size was {sz[0]}x{sz[1]})")
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path.name}: {e}")
        else:
            print(f"[FLAGGED] {file_path.name} (Size is {sz[0]}x{sz[1]})")

    print("\n" + "="*50)
    if auto_delete:
        print(f"[COMPLETE] Successfully deleted {len(outliers)} mis-sized images.")
    else:
        print("[DRY RUN COMPLETE] No files were actually deleted.")
        print("Change `auto_delete=True` at the bottom of the script to delete them.")
    print("="*50)


if __name__ == "__main__":
    DATASET_DIRECTORY = "dataset_60s"
    
    remove_wrong_sized_images(
        dataset_dir=DATASET_DIRECTORY, 
        auto_delete=True
    )
