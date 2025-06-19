from pathlib import Path
from PIL import Image # type: ignore

def verify_dataset(yaml_path):
    """Verify YOLO dataset structure and label formatting"""
    try:
        # Convert to Path object and get parent directory
        base_path = Path(yaml_path).parent
        print(f"🔍 Verifying dataset at: {base_path}")

        # Define paths
        images_train = base_path / "images/train"
        labels_train = base_path / "labels/train"

        # Verify paths exist
        assert images_train.exists(), f"Missing images directory: {images_train}"
        assert labels_train.exists(), f"Missing labels directory: {labels_train}"

        # Check image-label pairs
        for img_file in images_train.glob("*.jpg"):
            label_file = labels_train / f"{img_file.stem}.txt"
            assert label_file.exists(), f"Missing label: {label_file}"
            
            # Validate label format
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5, f"Invalid label format in {label_file} - expected 5 values"
                    cls, x, y, w, h = map(float, parts)
                    assert all(0 <= v <= 1 for v in [x, y, w, h]), f"Invalid coordinates in {label_file}"

        print("✅ Dataset verified successfully!")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

# Example usage - CHANGE THIS PATH TO YOUR ACTUAL data.yaml PATH
verify_dataset("/Users/prashantshah/Desktop/num_plates/data.yaml")
