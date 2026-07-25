from pathlib import Path

from PIL import Image


def scan_for_corrupted_images(dataset_dir: str):
    base_dir = Path(dataset_dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    print(f"Scanning {base_dir.resolve()} for corrupted images...\n")
    
    all_images = list(base_dir.rglob("*.png"))
    corrupted_files = []

    for img_path in all_images:
        # Check 1: Is the file completely empty (0 bytes)?
        if img_path.stat().st_size == 0:
            corrupted_files.append((img_path, "0-byte file"))
            continue
            
        # Check 2: Is the image data actually corrupted?
        try:
            with Image.open(img_path) as img:
                img.verify()  # Reads the header to ensure it's a valid PNG
        except Exception as e:
            corrupted_files.append((img_path, f"Unreadable format ({e})"))

    # Results
    if not corrupted_files:
        print(f"Success! Scanned {len(all_images)} images and found 0 corrupted files.")
    else:
        print(f"Found {len(corrupted_files)} corrupted files:\n")
        for file_path, reason in corrupted_files:
            print(f"- {file_path.name} ({reason})")
            
            # UNCOMMENT THE LINE BELOW TO AUTOMATICALLY DELETE THEM
            # file_path.unlink() 

if __name__ == "__main__":
    scan_for_corrupted_images("dataset")
