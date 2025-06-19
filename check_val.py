from pathlib import Path

val_labels = list(Path("labels/val").glob("*.txt"))
val_images = list(Path("images/val").glob("*"))

print(f"Validation images: {len(val_images)}")
print(f"Validation labels: {len(val_labels)}")
print("Missing labels:" if len(val_images) != len(val_labels) else "✅ Counts match")

for img in val_images:
    lbl = Path("labels/val") / f"{img.stem}.txt"
    if not lbl.exists():
        print(f"❌ {img.name} missing label")
