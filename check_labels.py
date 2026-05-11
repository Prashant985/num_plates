from pathlib import Path

dataset_path = Path("/Users/prashantshah/Desktop/num_plates")

def find_missing_plates():
    missing = []
    for label_file in (dataset_path/"labels/train").glob("*.txt"):
        if label_file.name == "classes.txt":
            continue
            
        with open(label_file) as f:
            if "0 " not in f.read():  # Check for plate class (0)
                img_name = label_file.stem + ".jpg"
                if (dataset_path/"images/train"/img_name).exists():
                    missing.append(f"{label_file.name} (image: {img_name})")
    return missing

print("Checking for missing plate annotations...")
missing = find_missing_plates()
print(f"\nFiles needing plates ({len(missing)}):")
print("\n".join(missing) if missing else "✅ All files have plate annotations!")
